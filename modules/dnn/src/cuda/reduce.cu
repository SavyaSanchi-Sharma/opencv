// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cub/block/block_reduce.cuh>

#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/reduce.hpp"

#include <opencv2/core.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace detail {
        constexpr int kMaxReduceRank = 16;

        // One block per output element; 256 threads (8 warps) hides memory latency on long reductions.
        constexpr int kReduceBlockSize = 256;

        // gridDim.x limit since cc 3.0; the kernels grid-stride over outputs, so no work is dropped.
        constexpr std::int64_t kMaxGridSize = 2147483647;

        static inline int reduce_grid_size(std::int64_t output_count) {
            return static_cast<int>(std::min(output_count, kMaxGridSize));
        }

        template <class T>
        struct accumulation_type { using type = T; };
        template <>
        struct accumulation_type<__half> { using type = float; };

        struct ReduceNdMetadata {
            int output_segment_count{};
            int reduction_segment_count{};
            std::int64_t output_segment_sizes[kMaxReduceRank]{};
            std::int64_t output_segment_strides[kMaxReduceRank]{};
            std::int64_t reduction_segment_sizes[kMaxReduceRank]{};
            std::int64_t reduction_segment_strides[kMaxReduceRank]{};
            std::int64_t output_count{};
            std::int64_t reduction_count{};
        };

        // Host-side: collapses `dims` into runs of reduced / non-reduced axes to shorten the per-element index decode.
        static inline void build_reduce_metadata(const std::vector<std::int64_t>& dims,
                                                 const std::vector<std::int64_t>& axes,
                                                 ReduceNdMetadata& metadata) {
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
        }

        __device__ __forceinline__ std::int64_t reduce_output_base(const ReduceNdMetadata& metadata, std::int64_t output_index) {
            std::int64_t remaining = output_index;
            std::int64_t base = 0;
            for (int segment = metadata.output_segment_count - 1; segment >= 0; --segment) {
                const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.output_segment_sizes[segment];
                if (segment != 0) remaining /= metadata.output_segment_sizes[segment];
                base += coordinate * metadata.output_segment_strides[segment];
            }
            return base;
        }

        __device__ __forceinline__ std::int64_t reduce_input_offset(const ReduceNdMetadata& metadata, std::int64_t input_base, std::int64_t reduction_index) {
            std::int64_t remaining = reduction_index;
            std::int64_t input_index = input_base;
            for (int segment = metadata.reduction_segment_count - 1; segment >= 0; --segment) {
                const std::int64_t coordinate = segment == 0 ? remaining : remaining % metadata.reduction_segment_sizes[segment];
                if (segment != 0) remaining /= metadata.reduction_segment_sizes[segment];
                input_index += coordinate * metadata.reduction_segment_strides[segment];
            }
            return input_index;
        }

        // Integer results follow saturate_cast<T>(double) in reduce2_layer.cpp: round half to even
        // (int64 rounds half away from zero, as round() does there), clamp to T, and NaN becomes T's minimum.
        template <typename T>
        __device__ __forceinline__ T store_result(double value, std::true_type) {
            using limits = ::cuda::std::numeric_limits<T>;
            if (isnan(value)) return limits::min();
            value = sizeof(T) == 8 ? round(value) : rint(value);
            if (value >= static_cast<double>(limits::max())) return limits::max();
            if (value <= static_cast<double>(limits::min())) return limits::min();
            return static_cast<T>(value);
        }

        // PROD over integers: the uint64_t accumulator holds the exact two's-complement product.
        template <typename T>
        __device__ __forceinline__ T store_result(std::uint64_t value, std::true_type) {
            using limits = ::cuda::std::numeric_limits<T>;
            const std::int64_t product = static_cast<std::int64_t>(value);
            if (product >= static_cast<std::int64_t>(limits::max())) return limits::max();
            if (product <= static_cast<std::int64_t>(limits::min())) return limits::min();
            return static_cast<T>(product);
        }

        template <typename T, typename TAccum>
        __device__ __forceinline__ T store_result(TAccum value, std::false_type) {
            return static_cast<T>(value);
        }

        template <typename T, typename TAccum>
        __device__ __forceinline__ T store_result(TAccum value) {
            return store_result<T>(value, std::is_integral<T>{});
        }

        // max/min accumulate no rounding error, so they keep T and track a running best value per thread.
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
        __global__ void reduce_minmax_nd_kernel(const T* input, T* output, ReduceNdMetadata metadata) {
            using State = MinMaxState<T, IsMax>;
            using BlockReduce = cub::BlockReduce<State, BlockSize>;
            __shared__ typename BlockReduce::TempStorage reduce_storage;
            __shared__ std::int64_t input_base;

            for (std::int64_t output_index = blockIdx.x;
                 output_index < metadata.output_count;
                 output_index += gridDim.x) {
                if (threadIdx.x == 0)
                    input_base = reduce_output_base(metadata, output_index);
                __syncthreads();

                State thread_state{};
                for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                     reduction_index += BlockSize)
                    thread_state.Add(input[reduce_input_offset(metadata, input_base, reduction_index)]);

                const State block_state = BlockReduce(reduce_storage).Reduce(thread_state, MergeMinMaxState<T, IsMax>{});
                if (threadIdx.x == 0) {
                    output[output_index] = block_state.Result();
                }
                __syncthreads();
            }
        }

        // Every op other than max/min differs only in transform, combine and finalisation,
        // so they share one kernel templated on the op.
        enum class GenericReduceOp { SUM, MEAN, PROD, L1, L2, SUM_SQUARE, LOG_SUM, LOG_SUM_EXP };

        // Integers accumulate in double like the CPU layer; PROD uses uint64_t so products stay exact up to 64 bits.
        template <typename T, GenericReduceOp Op>
        using generic_accum_t = std::conditional_t<!std::is_integral<T>::value, typename accumulation_type<T>::type,
                                std::conditional_t<Op == GenericReduceOp::PROD, std::uint64_t, double>>;

        template <typename TAccum>
        struct ProdCombine {
            __device__ __forceinline__ TAccum operator()(const TAccum& lhs, const TAccum& rhs) const { return lhs * rhs; }
        };

        template <typename TAccum>
        struct MaxCombine {
            __device__ __forceinline__ TAccum operator()(const TAccum& lhs, const TAccum& rhs) const { return lhs > rhs ? lhs : rhs; }
        };

        template <GenericReduceOp Op>
        struct GenericReduceStep;

        template <>
        struct GenericReduceStep<GenericReduceOp::SUM> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum&) { acc += value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&, std::int64_t) { return acc; }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::MEAN> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum&) { acc += value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&, std::int64_t count) { return acc / static_cast<TAccum>(count); }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::PROD> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum&) { acc *= value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&, std::int64_t) { return acc; }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::L1> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum&) { acc += fabs(value); }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&, std::int64_t) { return acc; }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::L2> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum&) { acc += value * value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&, std::int64_t) { return sqrt(acc); }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::SUM_SQUARE> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum&) { acc += value * value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&, std::int64_t) { return acc; }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::LOG_SUM> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum&) { acc += value; }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum&, std::int64_t) { return log(acc); }
        };

        template <>
        struct GenericReduceStep<GenericReduceOp::LOG_SUM_EXP> {
            template <typename TAccum>
            static __device__ __forceinline__ void Accumulate(TAccum& acc, TAccum value, const TAccum& shared_max) { acc += exp(value - shared_max); }
            template <typename TAccum>
            static __device__ __forceinline__ TAccum Finalize(TAccum acc, const TAccum& shared_max, std::int64_t) { return log(acc) + shared_max; }
        };

        template <typename T, GenericReduceOp Op, int BlockSize>
        __global__ void reduce_generic_nd_kernel(const T* input, T* output, ReduceNdMetadata metadata) {
            using TAccum = generic_accum_t<T, Op>;
            using Step = GenericReduceStep<Op>;
            using BlockReduce = cub::BlockReduce<TAccum, BlockSize>;
            __shared__ typename BlockReduce::TempStorage reduce_storage;
            __shared__ std::int64_t input_base;
            __shared__ TAccum shared_max;

            for (std::int64_t output_index = blockIdx.x;
                 output_index < metadata.output_count;
                 output_index += gridDim.x) {
                if (threadIdx.x == 0)
                    input_base = reduce_output_base(metadata, output_index);
                __syncthreads();

                // exp(x) overflows well before the reduction is done, so LOG_SUM_EXP
                // walks the reduction twice: once for max(x), then for sum(exp(x - max)).
                if (Op == GenericReduceOp::LOG_SUM_EXP) {
                    TAccum thread_max = -::cuda::std::numeric_limits<TAccum>::infinity();
                    for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                         reduction_index += BlockSize) {
                        const TAccum value = static_cast<TAccum>(input[reduce_input_offset(metadata, input_base, reduction_index)]);
                        thread_max = value > thread_max ? value : thread_max;
                    }
                    const TAccum block_max = BlockReduce(reduce_storage).Reduce(thread_max, MaxCombine<TAccum>{});
                    if (threadIdx.x == 0) shared_max = block_max;
                    __syncthreads();  // publishes shared_max and frees reduce_storage for the sum below
                }

                TAccum thread_acc = (Op == GenericReduceOp::PROD) ? TAccum(1) : TAccum(0);
                for (std::int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
                     reduction_index += BlockSize) {
                    const TAccum value = static_cast<TAccum>(input[reduce_input_offset(metadata, input_base, reduction_index)]);
                    Step::Accumulate(thread_acc, value, shared_max);
                }

                TAccum block_acc;
                if (Op == GenericReduceOp::PROD)
                    block_acc = BlockReduce(reduce_storage).Reduce(thread_acc, ProdCombine<TAccum>{});
                else
                    block_acc = BlockReduce(reduce_storage).Sum(thread_acc);

                if (threadIdx.x == 0)
                    output[output_index] = store_result<T>(Step::Finalize(block_acc, shared_max, metadata.reduction_count));
                __syncthreads();
            }
        }

        // Shared host side of every reduce op. cub::BlockReduce fixes the block size at compile time,
        // so the policy is built directly instead of through make_policy's occupancy query.
        template <class T, class Kernel>
        static void launch_reduce(Kernel kernel, const Stream& stream, Span<T> output, View<T> input,
                                  const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
            CV_Assert(dims.size() <= kMaxReduceRank);

            ReduceNdMetadata metadata;
            build_reduce_metadata(dims, axes, metadata);
            if (metadata.output_count == 0 || metadata.reduction_count == 0)
                return;

            const execution_policy policy(reduce_grid_size(metadata.output_count), kReduceBlockSize, stream);
            launch_kernel(kernel, policy, input.data().get(), output.data().get(), metadata);
        }

        template <class T, bool IsMax>
        static void reduce_minmax_nd(const Stream& stream, Span<T> output, View<T> input,
                                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
            launch_reduce<T>(reduce_minmax_nd_kernel<T, IsMax, kReduceBlockSize>, stream, output, input, dims, axes);
        }

        template <class T, GenericReduceOp Op>
        static void reduce_generic_nd(const Stream& stream, Span<T> output, View<T> input,
                                      const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
            launch_reduce<T>(reduce_generic_nd_kernel<T, Op, kReduceBlockSize>, stream, output, input, dims, axes);
        }
    }

    template <class T>
    void reduce_sum(const Stream& stream, Span<T> output, View<T> input,
                    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::SUM>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_mean(const Stream& stream, Span<T> output, View<T> input,
                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::MEAN>(stream, output, input, dims, axes);
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

    template <class T>
    void reduce_prod(const Stream& stream, Span<T> output, View<T> input,
                     const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::PROD>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_l1(const Stream& stream, Span<T> output, View<T> input,
                   const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::L1>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_l2(const Stream& stream, Span<T> output, View<T> input,
                   const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::L2>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_sum_square(const Stream& stream, Span<T> output, View<T> input,
                           const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::SUM_SQUARE>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_log_sum(const Stream& stream, Span<T> output, View<T> input,
                        const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::LOG_SUM>(stream, output, input, dims, axes);
    }

    template <class T>
    void reduce_log_sum_exp(const Stream& stream, Span<T> output, View<T> input,
                            const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes) {
        detail::reduce_generic_nd<T, detail::GenericReduceOp::LOG_SUM_EXP>(stream, output, input, dims, axes);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void reduce_sum(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
#endif
    template void reduce_sum(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_mean(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_max(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_min(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_prod(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l1(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_l2(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_sum_square(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void reduce_log_sum_exp(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
