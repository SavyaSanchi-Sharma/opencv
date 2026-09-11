// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "fast_divmod.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/max_pooling.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void max_pool_with_index(
            std::int64_t channels, std::int64_t height, std::int64_t width, std::int64_t depth,
            std::int64_t pooled_height, std::int64_t pooled_width, std::int64_t pooled_depth,
            std::int64_t kernel_h, std::int64_t kernel_w, std::int64_t kernel_d,
            std::int64_t stride_h, std::int64_t stride_w, std::int64_t stride_d,
            std::int64_t pad_h, std::int64_t pad_w, std::int64_t pad_d,
            std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t dilation_d,
            fast_divmod fdm_c, fast_divmod fdm_h, fast_divmod fdm_w, fast_divmod fdm_d,
            std::int64_t storage_order,
            View<T> input, size_type output_size, Span<T> output, std::int64_t* p_indices)
        {
            for (auto id : grid_stride_range(output_size)) {
                int d_index, w_index, h_index, c_index, n_index, id_tmp;
                fdm_d.divmod(id, id_tmp, d_index);
                fdm_w.divmod(id_tmp, id_tmp, w_index);
                fdm_h.divmod(id_tmp, id_tmp, h_index);
                fdm_c.divmod(id_tmp, n_index, c_index);

                std::int64_t d_start = static_cast<std::int64_t>(d_index) * stride_d - pad_d;
                std::int64_t w_start = static_cast<std::int64_t>(w_index) * stride_w - pad_w;
                std::int64_t h_start = static_cast<std::int64_t>(h_index) * stride_h - pad_h;

                std::int64_t d_end = csl::device::min<std::int64_t>(d_start + (kernel_d - 1) * dilation_d + 1, depth);
                std::int64_t w_end = csl::device::min<std::int64_t>(w_start + (kernel_w - 1) * dilation_w + 1, width);
                std::int64_t h_end = csl::device::min<std::int64_t>(h_start + (kernel_h - 1) * dilation_h + 1, height);

                d_start = csl::device::max<std::int64_t>(d_start, 0);
                w_start = csl::device::max<std::int64_t>(w_start, 0);
                h_start = csl::device::max<std::int64_t>(h_start, 0);

                std::int64_t d_index_max = -1;
                std::int64_t w_index_max = -1;
                std::int64_t h_index_max = -1;
                std::int64_t offset = ((static_cast<std::int64_t>(n_index) * channels + c_index) * height) * width * depth;
                const T* p_slice = input.data().get() + offset;
                T maxval = p_slice[(h_start * width + w_start) * depth + d_start] - static_cast<T>(1);
                for (std::int64_t d = d_start; d < d_end; d += dilation_d) {
                    for (std::int64_t w = w_start; w < w_end; w += dilation_w) {
                        for (std::int64_t h = h_start; h < h_end; h += dilation_h) {
                            std::int64_t pool_offset = (h * width + w) * depth + d;
                            if (p_slice[pool_offset] > maxval) {
                                h_index_max = h;
                                w_index_max = w;
                                d_index_max = d;
                                maxval = static_cast<float>(p_slice[pool_offset]);
                            }
                        }
                    }
                }
                output[id] = p_slice[(h_index_max * width + w_index_max) * depth + d_index_max];

                if (p_indices) {
                    p_indices[id] = storage_order == 0
                                        ? offset + h_index_max * width * depth + w_index_max * depth + d_index_max
                                        : offset + h_index_max + w_index_max * height + d_index_max * width * height;
                }
            }
        }
    }

    template <class T>
    void max_pool_with_index(const Stream& stream,
        Span<T> output, Span<std::int64_t> indices, View<T> input,
        const std::vector<std::int64_t>& input_shape,
        const std::vector<std::int64_t>& output_shape,
        const std::vector<std::int64_t>& kernel_shape,
        const std::vector<std::int64_t>& strides,
        const std::vector<std::int64_t>& pads,
        const std::vector<std::int64_t>& dilations,
        std::int64_t storage_order)
    {
        std::int64_t channels = input_shape[1];
        std::int64_t height = input_shape[2];
        std::int64_t width = kernel_shape.size() > 1 ? input_shape[3] : 1;
        std::int64_t depth = kernel_shape.size() > 2 ? input_shape[4] : 1;

        std::int64_t pooled_height = output_shape[2];
        std::int64_t pooled_width = kernel_shape.size() > 1 ? output_shape[3] : 1;
        std::int64_t pooled_depth = kernel_shape.size() > 2 ? output_shape[4] : 1;

        std::int64_t kernel_h = kernel_shape[0];
        std::int64_t kernel_w = kernel_shape.size() > 1 ? kernel_shape[1] : 1;
        std::int64_t kernel_d = kernel_shape.size() > 2 ? kernel_shape[2] : 1;
        std::int64_t stride_h = strides[0];
        std::int64_t stride_w = strides.size() > 1 ? strides[1] : 1;
        std::int64_t stride_d = strides.size() > 2 ? strides[2] : 1;
        std::int64_t pad_h = pads[0];
        std::int64_t pad_w = pads.size() >= 4 ? pads[1] : 0;
        std::int64_t pad_d = pads.size() == 6 ? pads[2] : 0;
        std::int64_t dilation_h = dilations[0];
        std::int64_t dilation_w = dilations.size() >= 2 ? dilations[1] : 1;
        std::int64_t dilation_d = dilations.size() == 3 ? dilations[2] : 1;

        std::int64_t output_size = 1;
        for (auto d : output_shape) output_size *= d;
        if (output_size == 0) return;

        fast_divmod fdm_c(static_cast<int>(channels));
        fast_divmod fdm_h(static_cast<int>(pooled_height));
        fast_divmod fdm_w(static_cast<int>(pooled_width));
        fast_divmod fdm_d(static_cast<int>(pooled_depth));

        std::int64_t* p_indices = indices.size() == 0 ? nullptr : indices.data().get();

        auto kernel = raw::max_pool_with_index<T>;
        auto policy = make_policy(kernel, static_cast<std::size_t>(output_size), 0, stream);
        launch_kernel(kernel, policy,
            channels, height, width, depth,
            pooled_height, pooled_width, pooled_depth,
            kernel_h, kernel_w, kernel_d,
            stride_h, stride_w, stride_d,
            pad_h, pad_w, pad_d,
            dilation_h, dilation_w, dilation_d,
            fdm_c, fdm_h, fdm_w, fdm_d,
            storage_order,
            input, static_cast<size_type>(output_size), output, p_indices);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void max_pool_with_index(const Stream&, Span<__half>, Span<std::int64_t>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::int64_t);
#endif
    template void max_pool_with_index(const Stream&, Span<float>, Span<std::int64_t>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::int64_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
