// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BROADCAST_COPY_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BROADCAST_COPY_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

/* Writes every element of `output` by wrapping its coordinate back into `input`'s
 * extent along each axis: out_coord[i] -> out_coord[i] % in_dims[i].
 *
 * That single mapping covers both ONNX ops that need it:
 *   Tile   -- out_dims[i] == in_dims[i] * repeats[i], so the modulo repeats the input.
 *   Expand -- broadcast axes have in_dims[i] == 1, so the modulo is always 0.
 *
 * `in_dims` and `out_dims` must have the same length (left-pad the input with 1s
 * first) and rank must not exceed kMaxBroadcastRank.
 */
constexpr int kMaxBroadcastRank = 8;

template <class T>
void broadcast_copy(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& in_dims,
    const std::vector<std::int64_t>& out_dims);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BROADCAST_COPY_HPP */
