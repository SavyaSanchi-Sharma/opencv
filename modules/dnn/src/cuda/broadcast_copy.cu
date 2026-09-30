// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "kernel_dispatcher.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/broadcast_copy.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        /* in_wrap[axis] is the input extent on Tile axes and 0 elsewhere; Expand axes carry a zero stride instead */
        template <class T, std::size_t Rank>
        __global__ void broadcast_copy(Span<T> output, View<T> input, array<size_type, Rank> out_strides,
                                       array<size_type, Rank> in_wrap, array<size_type, Rank> in_strides)
        {
            for (auto i : grid_stride_range(output.size())) {
                index_type remaining = i;
                index_type in_index = 0;
                for (int axis = 0; axis < Rank; axis++) {
                    index_type coord = remaining;
                    if (axis != Rank - 1) {
                        coord = remaining / out_strides[axis];
                        remaining -= coord * out_strides[axis];
                    }
                    if (in_wrap[axis])
                        coord %= in_wrap[axis];
                    in_index += coord * in_strides[axis];
                }
                output[i] = input[in_index];
            }
        }
    }

    template <class T, std::size_t Rank> static
    void launch_broadcast_copy(const Stream& stream, Span<T> output, View<T> input,
                               const std::vector<std::int64_t>& in_dims, const std::vector<std::int64_t>& out_dims)
    {
        array<size_type, Rank> out_strides, in_wrap, in_strides;
        size_type out_stride = 1, in_stride = 1;
        for (int axis = Rank - 1; axis >= 0; --axis) {
            const auto in_dim = static_cast<size_type>(in_dims[axis]);
            const auto out_dim = static_cast<size_type>(out_dims[axis]);
            out_strides[axis] = out_stride;
            in_strides[axis] = in_dim == 1 ? 0 : in_stride;
            in_wrap[axis] = (in_dim != 1 && in_dim != out_dim) ? in_dim : 0;
            out_stride *= out_dim;
            in_stride *= in_dim;
        }

        auto kernel = raw::broadcast_copy<T, Rank>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, out_strides, in_wrap, in_strides);
    }

    GENERATE_KERNEL_DISPATCHER(broadcast_copy_dispatcher, launch_broadcast_copy);

    template <class T>
    void broadcast_copy(const Stream& stream,
        Span<T> output, View<T> input,
        const std::vector<std::int64_t>& in_dims,
        const std::vector<std::int64_t>& out_dims)
    {
        CV_Assert(in_dims.size() == out_dims.size());
        CV_Assert(in_dims.size() <= kMaxBroadcastRank);

        for (std::size_t axis = 0; axis < in_dims.size(); axis++) {
            CV_Assert(in_dims[axis] > 0 && out_dims[axis] > 0);
            /* Tile repeats the axis and Expand broadcasts it; anything else is a shape
             * the caller should not have accepted. */
            CV_Assert(out_dims[axis] % in_dims[axis] == 0);
        }

        if (output.size() == 0)
            return;

        /* the CSL index types are 32-bit, like the rest of the CUDA backend */
        CV_Assert(output.size() <= static_cast<std::size_t>(std::numeric_limits<index_type>::max()));

        /* a scalar is copied as a rank 1 tensor of one element */
        if (in_dims.empty()) {
            broadcast_copy_dispatcher<T, 1, kMaxBroadcastRank>(1, stream, output, input,
                std::vector<std::int64_t>{1}, std::vector<std::int64_t>{1});
            return;
        }
        broadcast_copy_dispatcher<T, 1, kMaxBroadcastRank>(static_cast<int>(in_dims.size()), stream, output, input, in_dims, out_dims);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void broadcast_copy(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
#endif
    template void broadcast_copy(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<bool>, View<bool>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
