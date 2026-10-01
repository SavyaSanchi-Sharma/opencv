// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "vector_traits.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/softmax.hpp"

#include <opencv2/core.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace detail {
        // The register kernel keeps row_size / 32 values per thread; past 1024 elements they spill.
        constexpr int kWarpMaxRowSize = 1024;
        // The shared-memory warp kernel covers rows up to 2048, but only pays off with many rows.
        constexpr int kWarpSmemMaxRowSize = 2048;
        constexpr int kWarpSmemMinRows = 8192;
        // Neither warp kernel takes a row larger than this many bytes.
        constexpr int kWarpMaxRowBytes = 4096;
        // Block size of the register kernel: four warps per block.
        constexpr int kWarpKernelBlockSize = 128;
        // Hardware limit on threads per block.
        constexpr int kMaxBlockSize = 1024;

        // Rows of 128 or fewer share a warp two at a time; the kernel and launcher must agree on this.
        __host__ __device__ constexpr int rows_per_warp(int row_size_pow2) {
            return row_size_pow2 <= 128 ? 2 : 1;
        }

        template <class T>
        struct softmax_accum { using type = T; };
        template <>
        struct softmax_accum<__half> { using type = float; };

        template <typename T>
        struct SumOp {
            __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
        };

        template <typename T>
        struct MaxOp {
            __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? b : a; }
        };

        template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
        __device__ __forceinline__ void warp_reduce(acc_t* sum) {
            ReduceOp<acc_t> r;
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
                for (int i = 0; i < WARP_BATCH; ++i) {
                    acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
                    sum[i] = r(sum[i], b);
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
        __global__ void softmax_warp_kernel(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
            constexpr int next_power_of_two = 1 << log2_elements;
            constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
            constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
            constexpr int WARP_BATCH = rows_per_warp(next_power_of_two);

            int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

            int local_batches = batch_size - first_batch;
            if (local_batches > WARP_BATCH)
                local_batches = WARP_BATCH;

            int local_idx = threadIdx.x;

            src += first_batch * stride + local_idx;
            dst += first_batch * stride + local_idx;

            acc_t elements[WARP_BATCH][WARP_ITERATIONS];
            for (int i = 0; i < WARP_BATCH; ++i) {
                int batch_element_count = (i >= local_batches) ? 0 : element_count;
                for (int it = 0; it < WARP_ITERATIONS; ++it) {
                    int element_index = local_idx + it * WARP_SIZE;
                    if (element_index < batch_element_count) {
                        elements[i][it] = src[i * element_count + it * WARP_SIZE];
                    } else {
                        elements[i][it] = -::cuda::std::numeric_limits<acc_t>::infinity();
                    }
                }
            }

            acc_t max_value[WARP_BATCH];
#pragma unroll
            for (int i = 0; i < WARP_BATCH; ++i) {
                max_value[i] = elements[i][0];
#pragma unroll
                for (int it = 1; it < WARP_ITERATIONS; ++it) {
                    max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                }
            }
            warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, MaxOp>(max_value);

            acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
            for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
                for (int it = 0; it < WARP_ITERATIONS; ++it) {
                    if (is_log_softmax) {
                        sum[i] += std::exp((float)(elements[i][it] - max_value[i]));
                    } else {
                        elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
                        sum[i] += elements[i][it];
                    }
                }
            }
            warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, SumOp>(sum);

#pragma unroll
            for (int i = 0; i < WARP_BATCH; ++i) {
                if (i >= local_batches)
                    break;
                if (is_log_softmax) sum[i] = max_value[i] + std::log((float)(sum[i]));
#pragma unroll
                for (int it = 0; it < WARP_ITERATIONS; ++it) {
                    int element_index = local_idx + it * WARP_SIZE;
                    if (element_index < element_count) {
                        if (is_log_softmax) {
                            dst[i * element_count + it * WARP_SIZE] = elements[i][it] - sum[i];
                        } else {
                            dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
        __global__ void softmax_warp_smem_kernel(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
            constexpr int next_power_of_two = 1 << log2_elements;
            constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
            constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;

            int local_idx = threadIdx.x;
            src += blockIdx.x * stride + local_idx;
            dst += blockIdx.x * stride + local_idx;
            extern __shared__ unsigned char smem[];
            input_t(&elements)[WARP_ITERATIONS][WARP_SIZE] = *reinterpret_cast<input_t(*)[WARP_ITERATIONS][WARP_SIZE]>(smem);
#pragma unroll
            for (int it = 0; it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < element_count) {
                    elements[it][local_idx] = src[it * WARP_SIZE];
                } else {
                    static_assert(::cuda::std::numeric_limits<acc_t>::has_infinity,
                                  "type of acc_t should have infinity to avoid infinity function return 0");
                    elements[it][local_idx] = static_cast<input_t>(-::cuda::std::numeric_limits<acc_t>::infinity());
                }
            }
            input_t max_value = elements[0][local_idx];
#pragma unroll
            for (int it = 1; it < WARP_ITERATIONS; ++it) {
                max_value = (max_value > elements[it][local_idx]) ? max_value : elements[it][local_idx];
            }
            warp_reduce<input_t, 1, WARP_SIZE, MaxOp>(&max_value);
            acc_t sum{0.0f};
            for (int it = 0; it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index >= element_count)
                    break;
                if (is_log_softmax) {
                    sum += std::exp((float)(elements[it][local_idx] - max_value));
                } else {
                    acc_t tmp = std::exp((float)(elements[it][local_idx] - max_value));
                    elements[it][local_idx] = tmp;
                    sum += tmp;
                }
            }
            warp_reduce<acc_t, 1, WARP_SIZE, SumOp>(&sum);
            if (is_log_softmax) sum = static_cast<acc_t>(max_value) + std::log((float)(sum));
            acc_t invsum = static_cast<acc_t>(1.0f / sum);
#pragma unroll
            for (int it = 0; it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < element_count) {
                    if (is_log_softmax) {
                        dst[it * WARP_SIZE] = (float)elements[it][local_idx] - sum;
                    } else {
                        dst[it * WARP_SIZE] = (float)elements[it][local_idx] * invsum;
                    }
                } else {
                    break;
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        void launch_softmax_warp(const Stream& stream, output_t* dst, const input_t* src, int softmax_elements,
                                 int softmax_elements_stride, int batch_count) {
            if (softmax_elements == 0) {
                return;
            } else {
                int log2_elements = log2_ceil(softmax_elements);
                const int next_power_of_two = 1 << log2_elements;

                int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
                int threads_per_block, shared_memory_size;
                int batches_per_warp = rows_per_warp(next_power_of_two);
                if (next_power_of_two <= kWarpMaxRowSize) {
                    threads_per_block = kWarpKernelBlockSize;
                    shared_memory_size = 0;
                } else {
                    // One warp per block, with the whole row staged in shared memory.
                    threads_per_block = GPU_WARP_SIZE;
                    shared_memory_size = next_power_of_two * sizeof(input_t);
                }
                int warps_per_block = (threads_per_block / warp_size);
                int batches_per_block = warps_per_block * batches_per_warp;
                int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
                dim3 threads(warp_size, warps_per_block, 1);
                const execution_policy policy(blocks, threads, shared_memory_size, stream);

                static_assert(kWarpMaxRowSize == (1 << 10) && kWarpSmemMaxRowSize == (1 << 11),
                              "the switch below has one case per power of two up to kWarpSmemMaxRowSize");
                switch (log2_elements) {
#define LAUNCH_KERNEL(kernel_name, log2_elements_value)                      \
    launch_kernel(kernel_name<input_t, output_t, acc_t, log2_elements_value, is_log_softmax>, policy, \
                  dst, src, batch_count, softmax_elements_stride, softmax_elements);

#define CASE_LOG2_ELEMENTS(kernel_name, log2_elements_value)                      \
    case log2_elements_value: {                                                   \
        LAUNCH_KERNEL(kernel_name, log2_elements_value)                           \
    } break

                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 0);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 1);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 2);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 3);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 4);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 5);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 6);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 7);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 8);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 9);
                    CASE_LOG2_ELEMENTS(softmax_warp_kernel, 10);
                    CASE_LOG2_ELEMENTS(softmax_warp_smem_kernel, 11);
#undef LAUNCH_KERNEL
#undef CASE_LOG2_ELEMENTS
                    default:
                        CV_Error(cv::Error::StsInternal, "softmax: row is too long for the warp kernels");
                }
            }
        }

        dim3 pick_block_size(int ILP, uint64_t dim_size) {
            uint64_t block_size = 1;
            uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(kMaxBlockSize));

            if (ILP > 1) {
                max_block_size /= 2;
            }

            while (block_size < (max_block_size)) block_size *= 2;
            block_size = std::max(block_size, static_cast<uint64_t>(GPU_WARP_SIZE));
            return dim3(static_cast<unsigned int>(block_size));
        }

        template <typename T, typename AccumT>
        struct MaxStep {
            __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
                return ::max(max, (AccumT)v);
            }
        };

        template <typename T, typename AccumT>
        struct SumExpStep {
            __device__ __forceinline__ SumExpStep(AccumT v) : max_k(v) {}

            __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
                return sum + std::exp((AccumT)v - max_k);
            }

            const AccumT max_k;
        };

        template <template <typename> class Reduction, typename AccumT>
        __device__ __forceinline__ AccumT block_reduce(AccumT* smem, AccumT val,
                                                       const Reduction<AccumT>& r,
                                                       AccumT defaultVal) {
            __syncthreads();

            smem[threadIdx.x] = val;

            __syncthreads();

            AccumT warpVal = defaultVal;

            if (threadIdx.x < GPU_WARP_SIZE) {
                int warps_per_block = blockDim.x / GPU_WARP_SIZE;
                for (int i = 0; i < warps_per_block; ++i) {
                    warpVal = r(warpVal, smem[i * GPU_WARP_SIZE + threadIdx.x]);
                }
                smem[threadIdx.x] = warpVal;
            }

            __syncthreads();

            AccumT blockVal = defaultVal;

            if (threadIdx.x == 0) {
#pragma unroll
                for (int i = 0; i < GPU_WARP_SIZE; ++i) {
                    blockVal = r(blockVal, smem[i]);
                }
                smem[0] = blockVal;
            }

            __syncthreads();
            return smem[0];
        }

        template <template <typename, typename> class Reduction, int ILP, typename T, typename AccumT>
        __device__ __forceinline__ AccumT thread_reduce(int shift,
                                                        T* data,
                                                        int size,
                                                        const Reduction<T, AccumT>& r,
                                                        AccumT defaultVal) {
            using LoadT = aligned_vector<T, ILP>;
            AccumT threadVal = defaultVal;
            int offset = threadIdx.x;

            if (shift > 0) {
                data -= shift;
                size += shift;
                if (threadIdx.x >= shift && threadIdx.x < size) {
                    threadVal = r(threadVal, data[offset]);
                }
                size -= blockDim.x;
                data += blockDim.x;
            }

            if (size <= 0) return threadVal;

            int last = size % (ILP * blockDim.x);

            T v[ILP];
            LoadT* value = reinterpret_cast<LoadT*>(&v);

            for (; offset * ILP < (size - last); offset += blockDim.x) {
                *value = reinterpret_cast<LoadT*>(data)[offset];

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    threadVal = r(threadVal, v[j]);
                }
            }

            offset = size - last + threadIdx.x;
            for (; offset < size; offset += blockDim.x)
                threadVal = r(threadVal, data[offset]);

            return threadVal;
        }

        template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class OutputOp>
        __device__ __forceinline__ void write_output_vectorized(int size,
                                                                const int shift,
                                                                scalar_t* input,
                                                                outscalar_t* output,
                                                                OutputOp<scalar_t, accum_t, outscalar_t> output_op) {
            using LoadT = aligned_vector<scalar_t, ILP>;
            using StoreT = aligned_vector<outscalar_t, ILP>;

            int offset = threadIdx.x;

            if (shift > 0) {
                input -= shift;
                output -= shift;
                size += shift;

                if (threadIdx.x >= shift && threadIdx.x < size) {
                    output[offset] = output_op(input[offset]);
                }
                size -= blockDim.x;
                input += blockDim.x;
                output += blockDim.x;
            }

            if (size <= 0) return;

            const int last = size % (ILP * blockDim.x);

            scalar_t in_v[ILP];
            LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);

            outscalar_t out_v[ILP];
            StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);

            for (; offset * ILP < (size - last); offset += blockDim.x) {
                *in_value = reinterpret_cast<LoadT*>(input)[offset];

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    out_v[j] = output_op(in_v[j]);
                }

                reinterpret_cast<StoreT*>(output)[offset] = *out_value;
            }

            offset = size - last + threadIdx.x;
            for (; offset < size; offset += blockDim.x) {
                output[offset] = output_op(input[offset]);
            }
        }

        template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class OutputOp>
        __device__ __forceinline__ void write_output(int classes,
                                                     scalar_t* input,
                                                     outscalar_t* output,
                                                     OutputOp<scalar_t, accum_t, outscalar_t> output_op) {
            int offset = threadIdx.x;

            int last = classes % (ILP * blockDim.x);

            for (; offset < classes - last; offset += blockDim.x * ILP) {
                scalar_t tmp[ILP];

#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    tmp[j] = input[offset + j * blockDim.x];
                }
#pragma unroll
                for (int j = 0; j < ILP; ++j) {
                    output[offset + j * blockDim.x] = output_op(tmp[j]);
                }
            }

            for (; offset < classes; offset += blockDim.x) {
                output[offset] = output_op(input[offset]);
            }
        }

        template <typename T, typename AccumT, typename OutT>
        struct LogSoftmaxOutput {
            __device__ __forceinline__ LogSoftmaxOutput(AccumT max_input, AccumT sum)
                : max_input(max_input), logsum(std::log(sum)) {}

            __device__ __forceinline__ OutT operator()(T input) const {
                return static_cast<OutT>((AccumT)input - max_input - logsum);
            }

            const AccumT max_input;
            const AccumT logsum;
        };

        template <typename T, typename AccumT, typename OutT>
        struct SoftmaxOutput {
            __device__ __forceinline__ SoftmaxOutput(AccumT max_input, AccumT sum)
                : max_input(max_input), sum(sum) {}

            __device__ __forceinline__ OutT operator()(T input) const {
                return static_cast<OutT>(std::exp((AccumT)input - max_input) / sum);
            }

            const AccumT max_input;
            const AccumT sum;
        };

        template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t,
                  template <typename, typename, typename> class OutputOp>
        __global__ void softmax_block_kernel(outscalar_t* output, scalar_t* input, int classes,
                                             int input_stride, int output_stride) {
            extern __shared__ unsigned char smem[];
            auto sdata = reinterpret_cast<accscalar_t*>(smem);

            input += blockIdx.x * input_stride;
            output += blockIdx.x * output_stride;

            const int input_align_bytes = ILP * sizeof(scalar_t);
            const int output_align_bytes = ILP * sizeof(outscalar_t);

            const int shift = ((uint64_t)input) % input_align_bytes / sizeof(scalar_t);
            const int output_shift = ((uint64_t)output) % output_align_bytes / sizeof(outscalar_t);

            accscalar_t threadMax = thread_reduce<MaxStep, ILP, scalar_t, accscalar_t>(
                shift, input, classes, MaxStep<scalar_t, accscalar_t>(), -::cuda::std::numeric_limits<accscalar_t>::max());
            accscalar_t max_k = block_reduce<MaxOp, accscalar_t>(
                sdata, threadMax, MaxOp<accscalar_t>(), -::cuda::std::numeric_limits<accscalar_t>::max());

            accscalar_t threadExp = thread_reduce<SumExpStep, ILP, scalar_t, accscalar_t>(
                shift, input, classes, SumExpStep<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
            accscalar_t sumAll = block_reduce<SumOp, accscalar_t>(
                sdata, threadExp, SumOp<accscalar_t>(), static_cast<accscalar_t>(0));

            OutputOp<scalar_t, accscalar_t, outscalar_t> output_op(max_k, sumAll);

            if (shift == output_shift) {
                write_output_vectorized<ILP, scalar_t, accscalar_t, outscalar_t, OutputOp>(classes, shift, input, output, output_op);
            } else {
                write_output<ILP, scalar_t, accscalar_t, outscalar_t, OutputOp>(classes, input, output, output_op);
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        void launch_softmax_block(const Stream& stream, output_t* output, const input_t* input, int softmax_elements,
                                  int input_stride, int output_stride, int batch_count) {
            dim3 grid(batch_count);
            // Each thread loads 16 bytes at a time: 4 floats or 8 halves.
            constexpr int ILP = sizeof(float4) / sizeof(input_t);
            dim3 block = pick_block_size(ILP, softmax_elements);
            const execution_policy policy(grid, block, block.x * sizeof(acc_t), stream);
            if (is_log_softmax) {
                launch_kernel(softmax_block_kernel<ILP, input_t, acc_t, output_t, LogSoftmaxOutput>, policy,
                              output, const_cast<input_t*>(input), softmax_elements, input_stride, output_stride);
            } else {
                launch_kernel(softmax_block_kernel<ILP, input_t, acc_t, output_t, SoftmaxOutput>, policy,
                              output, const_cast<input_t*>(input), softmax_elements, input_stride, output_stride);
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        __global__ void softmax_strided_kernel(output_t* dst, const input_t* src,
                                               int axis_size, int inner_size, std::int64_t num_lanes) {
            for (std::int64_t lane = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 lane < num_lanes;
                 lane += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
            {
                const std::int64_t outer = lane / inner_size;
                const std::int64_t stride = inner_size;
                const std::int64_t base = outer * static_cast<std::int64_t>(axis_size) * stride + (lane - outer * stride);

                acc_t max_value = -::cuda::std::numeric_limits<acc_t>::infinity();
                for (int k = 0; k < axis_size; k++) {
                    const acc_t v = static_cast<acc_t>(src[base + k * stride]);
                    max_value = v > max_value ? v : max_value;
                }

                acc_t sum = static_cast<acc_t>(0);
                for (int k = 0; k < axis_size; k++)
                    sum += std::exp((float)(static_cast<acc_t>(src[base + k * stride]) - max_value));

                if (is_log_softmax) {
                    const acc_t log_sum = static_cast<acc_t>(std::log((float)sum));
                    for (int k = 0; k < axis_size; k++) {
                        const std::int64_t o = base + k * stride;
                        dst[o] = static_cast<output_t>(static_cast<acc_t>(src[o]) - max_value - log_sum);
                    }
                } else {
                    for (int k = 0; k < axis_size; k++) {
                        const std::int64_t o = base + k * stride;
                        dst[o] = static_cast<output_t>(std::exp((float)(static_cast<acc_t>(src[o]) - max_value)) / sum);
                    }
                }
            }
        }

        template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
        void launch_softmax_strided(const Stream& stream, output_t* output, const input_t* input,
                                    int axis_size, int outer_size, int inner_size) {
            const std::int64_t num_lanes = static_cast<std::int64_t>(outer_size) * static_cast<std::int64_t>(inner_size);
            auto kernel = softmax_strided_kernel<input_t, output_t, acc_t, is_log_softmax>;
            // The kernel grid-strides over lanes, so the grid only needs to fill the GPU, not cover every lane.
            auto policy = make_policy(kernel, static_cast<std::size_t>(num_lanes), 0, stream);
            launch_kernel(kernel, policy, output, input, axis_size, inner_size, num_lanes);
        }
    }

    template <class T>
    void softmax(const Stream& stream, Span<T> output, View<T> input, int axis_size, int outer_size, int inner_size, bool log_softmax) {
        using acc_t = typename detail::softmax_accum<T>::type;

        const int D = axis_size;
        const int N = outer_size;
        if (D == 0 || N == 0 || inner_size == 0)
            return;

        T* dst = output.data().get();
        const T* src = input.data().get();

        if (inner_size != 1) {
            if (log_softmax)
                detail::launch_softmax_strided<T, T, acc_t, true>(stream, dst, src, D, N, inner_size);
            else
                detail::launch_softmax_strided<T, T, acc_t, false>(stream, dst, src, D, N, inner_size);
            return;
        }

        const int row_bytes = D * static_cast<int>(sizeof(T));
        const bool fits_warp = D <= detail::kWarpMaxRowSize && row_bytes <= detail::kWarpMaxRowBytes;
        const bool fits_warp_smem = D > detail::kWarpMaxRowSize && D <= detail::kWarpSmemMaxRowSize &&
                                    row_bytes <= detail::kWarpMaxRowBytes && N >= detail::kWarpSmemMinRows;
        if (fits_warp || fits_warp_smem) {
            if (log_softmax)
                detail::launch_softmax_warp<T, T, acc_t, true>(stream, dst, src, D, D, N);
            else
                detail::launch_softmax_warp<T, T, acc_t, false>(stream, dst, src, D, D, N);
        } else {
            if (log_softmax)
                detail::launch_softmax_block<T, T, acc_t, true>(stream, dst, src, D, D, D, N);
            else
                detail::launch_softmax_block<T, T, acc_t, false>(stream, dst, src, D, D, D, N);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void softmax(const Stream&, Span<__half>, View<__half>, int, int, int, bool);
#endif
    template void softmax(const Stream&, Span<float>, View<float>, int, int, int, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
