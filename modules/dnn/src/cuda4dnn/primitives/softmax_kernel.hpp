// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_KERNEL_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_KERNEL_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/softmax.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class SoftmaxKernelOp final : public CUDABackendNode {
    public:
        SoftmaxKernelOp(csl::Stream stream_, int axis_, bool log_softmax_)
            : stream(std::move(stream_)), axis(axis_), log_softmax(log_softmax_)
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            const int rank = inShape.dims;
            const int ax = normalize_axis(axis, rank);

            const int axis_size = inShape[ax];
            int inner_size = 1;
            for (int i = ax + 1; i < rank; i++)
                inner_size *= inShape[i];
            CV_CheckEQ(inner_size, 1, "CUDA softmax (JIT) supports softmax over the last axis only");

            int outer_size = 1;
            for (int i = 0; i < ax; i++)
                outer_size *= inShape[i];

            kernels::softmax<T>(stream, output, input, axis_size, outer_size, log_softmax);
        }

    private:
        csl::Stream stream;
        int axis;
        bool log_softmax;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_KERNEL_HPP */
