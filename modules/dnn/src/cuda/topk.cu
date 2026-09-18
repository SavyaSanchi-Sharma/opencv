// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cub/block/block_reduce.cuh>

#include "types.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/topk.hpp"

#include <opencv2/core.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        struct Candidate {
            float value;
            int index;
        };

        /* Strict weak ordering matching topk2_layer.cpp's comparators: better value
         * first, and on an exact tie the smaller index. */
        __device__ __forceinline__ bool better_than(const Candidate& a, const Candidate& b, bool largest) {
            if (a.value == b.value)
                return a.index < b.index;
            return largest ? (a.value > b.value) : (a.value < b.value);
        }

        struct MergeCandidate {
            bool largest;
            __device__ __forceinline__ Candidate operator()(const Candidate& a, const Candidate& b) const {
                return better_than(a, b, largest) ? a : b;
            }
        };

        template <class T, int BlockSize>
        __global__ void topk(Span<T> values, Span<std::int64_t> indices, View<T> input,
                             int outer, int dim_axis, int inner, int k, bool largest)
        {
            using BlockReduce = cub::BlockReduce<Candidate, BlockSize>;
            __shared__ typename BlockReduce::TempStorage reduce_storage;
            __shared__ Candidate selected;

            const int nslices = outer * inner;
            const float worst = largest ? -::cuda::std::numeric_limits<float>::infinity()
                                        :  ::cuda::std::numeric_limits<float>::infinity();

            for (int slice = blockIdx.x; slice < nslices; slice += gridDim.x) {
                const int b = slice / inner;
                const int j = slice - b * inner;

                const std::int64_t in_base  = static_cast<std::int64_t>(b) * dim_axis * inner + j;
                const std::int64_t out_base = static_cast<std::int64_t>(b) * k * inner + j;

                for (int rank = 0; rank < k; ++rank) {
                    Candidate best{worst, ::cuda::std::numeric_limits<int>::max()};

                    for (int u = threadIdx.x; u < dim_axis; u += BlockSize) {
                        const float v = static_cast<float>(input.data().get()[in_base + static_cast<std::int64_t>(u) * inner]);
                        const Candidate cand{v, u};

                        /* Everything at or before the previous pick is spent. Ordering
                         * strictly means no element can be chosen twice, and duplicates
                         * are still all reachable because index breaks the tie. */
                        if (rank > 0 && !better_than(selected, cand, largest))
                            continue;

                        if (better_than(cand, best, largest))
                            best = cand;
                    }

                    const Candidate winner = BlockReduce(reduce_storage).Reduce(best, MergeCandidate{largest});

                    if (threadIdx.x == 0) {
                        selected = winner;
                        const std::int64_t out = out_base + static_cast<std::int64_t>(rank) * inner;
                        values[out] = static_cast<T>(winner.value);
                        indices[out] = static_cast<std::int64_t>(winner.index);
                    }
                    __syncthreads();  // publishes `selected` and frees reduce_storage for the next rank
                }
            }
        }
    }

    template <class T>
    void topk(const Stream& stream,
        Span<T> values, Span<std::int64_t> indices, View<T> input,
        int outer, int dim_axis, int inner, int k, bool largest)
    {
        CV_Assert(outer > 0 && dim_axis > 0 && inner > 0);
        CV_Assert(k > 0 && k <= dim_axis);

        constexpr int block_size = 256;
        constexpr int max_blocks = 65535;
        const int nslices = outer * inner;
        const int grid_size = std::min(max_blocks, nslices);

        raw::topk<T, block_size><<<grid_size, block_size, 0, stream.get()>>>(
            values, indices, input, outer, dim_axis, inner, k, largest);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void topk(const Stream&, Span<__half>, Span<std::int64_t>, View<__half>, int, int, int, int, bool);
#endif
    template void topk(const Stream&, Span<float>, Span<std::int64_t>, View<float>, int, int, int, int, bool);
    template void topk(const Stream&, Span<int8_t>, Span<std::int64_t>, View<int8_t>, int, int, int, int, bool);
    template void topk(const Stream&, Span<uint8_t>, Span<std::int64_t>, View<uint8_t>, int, int, int, int, bool);
    template void topk(const Stream&, Span<int32_t>, Span<std::int64_t>, View<int32_t>, int, int, int, int, bool);
    template void topk(const Stream&, Span<int64_t>, Span<std::int64_t>, View<int64_t>, int, int, int, int, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
