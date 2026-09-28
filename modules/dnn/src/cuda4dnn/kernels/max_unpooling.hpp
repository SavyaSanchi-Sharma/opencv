// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MAX_UNPOOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MAX_UNPOOLING_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include <cstddef>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    /* empty indices are skipped; global_indices and column_major give ONNX MaxPool Indices */
    template <class T, class T_INDEX>
    void max_pooling_with_indices(
        const csl::Stream& stream,
        csl::TensorSpan<T> output, csl::TensorSpan<T_INDEX> indices, csl::TensorView<T> input,
        const std::vector<std::size_t>& kernel_size, const std::vector<std::size_t>& strides,
        const std::vector<std::size_t>& padding_left, const std::vector<std::size_t>& dilations = {},
        bool global_indices = false, bool column_major = false);

    template <class T, class T_INDEX>
    void max_unpooling(
        const csl::Stream& stream,
        csl::TensorSpan<T> output, csl::TensorView<T> input, csl::TensorView<T_INDEX> indices,
        const std::vector<std::size_t>& window_size, const std::vector<std::size_t>& strides,
        const std::vector<std::size_t>& padding_left);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MAX_UNPOOLING_HPP */
