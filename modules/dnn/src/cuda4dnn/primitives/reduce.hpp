// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_REDUCE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_REDUCE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/reduce.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class ReduceOpType { SUM, MEAN, MAX, MIN };

    template <class T>
    class ReduceOp final : public CUDABackendNode {
    public:
        ReduceOp(csl::Stream stream_, ReduceOpType op_, std::vector<std::int64_t> axes_)
            : stream(std::move(stream_)), op(op_), axes(std::move(axes_))
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            std::vector<std::int64_t> dims(inShape.begin(), inShape.end());

            switch (op) {
                case ReduceOpType::MEAN: kernels::reduce_mean<T>(stream, output, input, dims, axes); break;
                case ReduceOpType::SUM:  kernels::reduce_sum<T>(stream, output, input, dims, axes);  break;
                case ReduceOpType::MAX:  kernels::reduce_max<T>(stream, output, input, dims, axes);  break;
                case ReduceOpType::MIN:  kernels::reduce_min<T>(stream, output, input, dims, axes);  break;
            }
        }

    private:
        csl::Stream stream;
        ReduceOpType op;
        std::vector<std::int64_t> axes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_REDUCE_HPP */
