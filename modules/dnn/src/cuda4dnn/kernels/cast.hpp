// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CAST_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CAST_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

// Only the (source, target) pairs this engine's ONNX graphs actually need are
// implemented -- add more pairs as they're actually needed rather than
// building a general conversion matrix.
void cast_int64_to_fp32(const csl::Stream& stream, csl::Span<float> output, csl::View<std::int64_t> input);
void cast_fp32_to_int64(const csl::Stream& stream, csl::Span<std::int64_t> output, csl::View<float> input);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CAST_HPP */
