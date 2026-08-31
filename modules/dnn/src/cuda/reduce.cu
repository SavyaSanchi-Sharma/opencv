// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cub/block/block_reduce.cuh>

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/reduce.hpp"

#include <opencv2/core.hpp>

#include <cuda/std/limits>

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace detail {
        constexpr int kMaxReduceRank = 16;

        template <class T>
        struct accumulation_type { using type = T; };
        template <>
        struct accumulation_type<__half> { using type = float; };

        struct ReduceSumNdMetadata {
            int output_segment_count{};
            int reduction_segment_count{};
            std::int64_t output_segment_sizes[kMaxReduceRank]{};
            std::int64_t output_segment_strides[kMaxReduceRank]{};
            std::int64_t reduction_segment_sizes[kMaxReduceRank]{};
            std::int64_t reduction_segment_strides[kMaxReduceRank]{};
            std::int64_t output_count{};
            std::int64_t reduction_count{};
        };

        template <typename T>
        struct SumState {
            T sum{};

            __device__ __forceinline__ void Add(T value) { sum += value; }
            __device__ __forceinline__ T Result() const { return sum; }
        };

        template <>
        struct SumState<double> {
            double sum{};
            double correction{};

            __device__ __forceinline__ void Add(double value) {
                const double next = __dadd_rn(sum, value);
                const double error = fabs(sum) >= fabs(value)
                                         ? __dadd_rn(__dsub_rn(sum, next), value)
                                         : __dadd_rn(__dsub_rn(value, next), sum);
                correction = __dadd_rn(correction, error);
                sum = next;
            }

            __device__ __forceinline__ double Result() const { return __dadd_rn(sum, correction); }
        };

        template <typename T>
        struct MergeSumState {
            __device__ __forceinline__ SumState<T> operator()(SumState<T> lhs, const SumState<T>& rhs) const {
                lhs.Add(rhs.sum);
                if constexpr (std::is_same_v<T, double>) {
                    lhs.Add(rhs.correction);
                }
                return lhs;
            }
        };

        template <typename T, typename TAccum>
        __device__ __forceinline__ T CastReduceSumResult(TAccum value) {
            if constexpr (std::is_integral_v<T>) {
                const double value_as_double = static_cast<double>(value);
                const double max_value = static_cast<double>(::cuda::std::numeric_limits<T>::max());
                const double min_value = static_cast<double>(::cuda::std::numeric_limits<T>::min());
                if (value_as_double >= max_value) return ::cuda::std::numeric_limits<T>::max();
                if (value_as_double <= min_value) return ::cuda::std::numeric_limits<T>::min();
            }
            return static_cast<T>(value);
        }

        template <typename T, int BlockSize>
        __global__ void reduce_sum_nd_kernel(const T* input, T* output, ReduceSumNdMetadata metadata, double inv_norm) {
            using TAccum = std::conditional_t<std::is_integral_v<T>, double, typename accumulation_type<T>::type>;
            using BlockReduce = cub::BlockReduce<SumState<TAccum>, BlockSize>;
            __shared__ typename BlockReduce::TempStorage reduce_storage;
            __shared__ std::int64_t input_base;

            for (std::int64_t output_index = blockIdx.x;
                 output_index < metadata.output_count;
                 output_index += gridDim.x) {
                if (threadIdx.x == 0) {
                    std::int64_t remaining = output_index;
                    std::int64_t base = 0;
                    for (int segment = metadata.output_segment_count - 1; segment >= 0; --segment) {
                        const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.output_segment_sizes[segment];
                        if (segment != 0) remaining /= metadata.output_segment_sizes[segment];
                        base += coordinate * metadata.output_segment_strides[segment];
                    }
                    input_base = base;
                }
                __syncthreads();

                SumState<TAccum> thread_sum{};
                for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                     reduction_index += BlockSize) {
                    std::int64_t remaining = reduction_index;
                    std::int64_t input_index = input_base;
                    for (int segment = metadata.reduction_segment_count - 1; segment >= 0; --segment) {
                        const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.reduction_segment_sizes[segment];
                        if (segment != 0) remaining /= metadata.reduction_segment_sizes[segment];
                        input_index += coordinate * metadata.reduction_segment_strides[segment];
                    }
                    thread_sum.Add(static_cast<TAccum>(input[input_index]));
                }

                const SumState<TAccum> block_sum = BlockReduce(reduce_storage).Reduce(thread_sum, MergeSumState<TAccum>{});
                if (threadIdx.x == 0) {
                    output[output_index] = CastReduceSumResult<T>(static_cast<TAccum>(block_sum.Result() * static_cast<TAccum>(inv_norm)));
                }
                __syncthreads();
            }
        }

        template <class T>
        static void reduce_nd(const Stream& stream, Span<T> output, View<T> input,
                              const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes, bool mean) {
            CV_Assert(dims.size() <= kMaxReduceRank);

            ReduceSumNdMetadata metadata;
            const int rank = static_cast<int>(dims.size());
            std::array<std::int64_t, kMaxReduceRank> strides{};
            std::int64_t stride = 1;
            for (int axis = rank - 1; axis >= 0; --axis) {
                CV_Assert(dims[axis] > 0);
                strides[axis] = stride;
                stride *= dims[axis];
            }

            std::array<bool, kMaxReduceRank> reduced{};
            if (axes.empty()) {
                for (int axis = 0; axis < rank; ++axis) reduced[axis] = true;
            } else {
                for (std::int64_t axis : axes) {
                    if (axis < 0) axis += rank;
                    CV_Assert(axis >= 0 && axis < rank);
                    reduced[axis] = true;
                }
            }

            std::int64_t output_count = 1;
            std::int64_t reduction_count = 1;
            for (int axis = 0; axis < rank;) {
                const bool is_reduced = reduced[axis];
                std::int64_t segment_size = 1;
                int last_axis = axis;
                do {
                    segment_size *= dims[axis];
                    last_axis = axis++;
                } while (axis < rank && reduced[axis] == is_reduced);

                if (is_reduced) {
                    const int segment = metadata.reduction_segment_count++;
                    metadata.reduction_segment_sizes[segment] = segment_size;
                    metadata.reduction_segment_strides[segment] = strides[last_axis];
                    reduction_count *= segment_size;
                } else {
                    const int segment = metadata.output_segment_count++;
                    metadata.output_segment_sizes[segment] = segment_size;
                    metadata.output_segment_strides[segment] = strides[last_axis];
                    output_count *= segment_size;
                }
            }
            metadata.output_count = output_count;
            metadata.reduction_count = reduction_count;

            if (output_count == 0 || reduction_count == 0)
                return;

            const double inv_norm = mean ? 1.0 / static_cast<double>(reduction_count) : 1.0;

            constexpr int block_size = 256;
            constexpr int max_blocks = 65535;
            const int grid_size = static_cast<int>(std::min<std::int64_t>(max_blocks, output_count));
            reduce_sum_nd_kernel<T, block_size><<<grid_size, block_size, 0, stream.get()>>>(
                input.data().get(), output.data().get(), metadata, inv_norm);
        }
    }

    template <class T>
    void reduce_sum(const Stream& stream, Span<T> output, View<T> input,
                    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_nd<T>(stream, output, input, dims, axes, false);
    }

    template <class T>
    void reduce_mean(const Stream& stream, Span<T> output, View<T> input,
                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_nd<T>(stream, output, input, dims, axes, true);
    }

    namespace detail {
        // Independent from ReduceSumNdMetadata's SUM/MEAN kernel above -- max/min don't
        // accumulate floating point error, so no TAccum promotion or normalization is
        // needed, just a running best-so-far value per thread.
        template <typename T, bool IsMax>
        struct MinMaxState {
            T value{};
            bool has_value{false};

            __device__ __forceinline__ void Add(T v) {
                if (!has_value || (IsMax ? (v > value) : (v < value))) {
                    value = v;
                    has_value = true;
                }
            }
            __device__ __forceinline__ T Result() const { return value; }
        };

        template <typename T, bool IsMax>
        struct MergeMinMaxState {
            __device__ __forceinline__ MinMaxState<T, IsMax> operator()(MinMaxState<T, IsMax> lhs, const MinMaxState<T, IsMax>& rhs) const {
                if (rhs.has_value) lhs.Add(rhs.value);
                return lhs;
            }
        };

        template <typename T, bool IsMax, int BlockSize>
        __global__ void reduce_minmax_nd_kernel(const T* input, T* output, ReduceSumNdMetadata metadata) {
            using State = MinMaxState<T, IsMax>;
            using BlockReduce = cub::BlockReduce<State, BlockSize>;
            __shared__ typename BlockReduce::TempStorage reduce_storage;
            __shared__ std::int64_t input_base;

            for (std::int64_t output_index = blockIdx.x;
                 output_index < metadata.output_count;
                 output_index += gridDim.x) {
                if (threadIdx.x == 0) {
                    std::int64_t remaining = output_index;
                    std::int64_t base = 0;
                    for (int segment = metadata.output_segment_count - 1; segment >= 0; --segment) {
                        const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.output_segment_sizes[segment];
                        if (segment != 0) remaining /= metadata.output_segment_sizes[segment];
                        base += coordinate * metadata.output_segment_strides[segment];
                    }
                    input_base = base;
                }
                __syncthreads();

                State thread_state{};
                for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                     reduction_index += BlockSize) {
                    std::int64_t remaining = reduction_index;
                    std::int64_t input_index = input_base;
                    for (int segment = metadata.reduction_segment_count - 1; segment >= 0; --segment) {
                        const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.reduction_segment_sizes[segment];
                        if (segment != 0) remaining /= metadata.reduction_segment_sizes[segment];
                        input_index += coordinate * metadata.reduction_segment_strides[segment];
                    }
                    thread_state.Add(input[input_index]);
                }

                const State block_state = BlockReduce(reduce_storage).Reduce(thread_state, MergeMinMaxState<T, IsMax>{});
                if (threadIdx.x == 0) {
                    output[output_index] = block_state.Result();
                }
                __syncthreads();
            }
        }

        template <class T, bool IsMax>
        static void reduce_minmax_nd(const Stream& stream, Span<T> output, View<T> input,
                                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
            CV_Assert(dims.size() <= kMaxReduceRank);

            ReduceSumNdMetadata metadata;
            const int rank = static_cast<int>(dims.size());
            std::array<std::int64_t, kMaxReduceRank> strides{};
            std::int64_t stride = 1;
            for (int axis = rank - 1; axis >= 0; --axis) {
                CV_Assert(dims[axis] > 0);
                strides[axis] = stride;
                stride *= dims[axis];
            }

            std::array<bool, kMaxReduceRank> reduced{};
            if (axes.empty()) {
                for (int axis = 0; axis < rank; ++axis) reduced[axis] = true;
            } else {
                for (std::int64_t axis : axes) {
                    if (axis < 0) axis += rank;
                    CV_Assert(axis >= 0 && axis < rank);
                    reduced[axis] = true;
                }
            }

            std::int64_t output_count = 1;
            std::int64_t reduction_count = 1;
            for (int axis = 0; axis < rank;) {
                const bool is_reduced = reduced[axis];
                std::int64_t segment_size = 1;
                int last_axis = axis;
                do {
                    segment_size *= dims[axis];
                    last_axis = axis++;
                } while (axis < rank && reduced[axis] == is_reduced);

                if (is_reduced) {
                    const int segment = metadata.reduction_segment_count++;
                    metadata.reduction_segment_sizes[segment] = segment_size;
                    metadata.reduction_segment_strides[segment] = strides[last_axis];
                    reduction_count *= segment_size;
                } else {
                    const int segment = metadata.output_segment_count++;
                    metadata.output_segment_sizes[segment] = segment_size;
                    metadata.output_segment_strides[segment] = strides[last_axis];
                    output_count *= segment_size;
                }
            }
            metadata.output_count = output_count;
            metadata.reduction_count = reduction_count;

            if (output_count == 0 || reduction_count == 0)
                return;

            constexpr int block_size = 256;
            constexpr int max_blocks = 65535;
            const int grid_size = static_cast<int>(std::min<std::int64_t>(max_blocks, output_count));
            reduce_minmax_nd_kernel<T, IsMax, block_size><<<grid_size, block_size, 0, stream.get()>>>(
                input.data().get(), output.data().get(), metadata);
        }
    }

    template <class T>
    void reduce_max(const Stream& stream, Span<T> output, View<T> input,
                    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_minmax_nd<T, true>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_min(const Stream& stream, Span<T> output, View<T> input,
                    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_minmax_nd<T, false>(stream, output, input, dims, axes);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void reduce_sum(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
#endif
    template void reduce_sum(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
