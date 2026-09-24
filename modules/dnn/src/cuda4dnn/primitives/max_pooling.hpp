// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MAX_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MAX_POOLING_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/max_pooling.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct MaxPoolConfiguration {
        std::vector<std::int64_t> kernel_shape;
        std::vector<std::int64_t> strides;
        std::vector<std::int64_t> pads;
        std::vector<std::int64_t> dilations;
        std::int64_t storage_order;
    };

    template <class T>
    class MaxPoolOp final : public CUDABackendNode {
    public:
        MaxPoolOp(csl::Stream stream_, const MaxPoolConfiguration& config)
            : stream(std::move(stream_)),
              kernel_shape(config.kernel_shape),
              strides(config.strides),
              pads(config.pads),
              dilations(config.dilations),
              storage_order(config.storage_order)
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() >= 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            MatShape outShape = cv::dnn::shape(outputs[0]);
            std::vector<std::int64_t> in_shape(inShape.begin(), inShape.end());
            std::vector<std::int64_t> out_shape(outShape.begin(), outShape.end());

            csl::Span<std::int64_t> indices;
            if (outputs.size() > 1)
                indices = csl::spanOf<std::int64_t>(outputs[1]);

            kernels::max_pool_with_index<T>(stream, output, indices, input,
                                            in_shape, out_shape, kernel_shape, strides, pads, dilations, storage_order);
        }

    private:
        csl::Stream stream;
        std::vector<std::int64_t> kernel_shape, strides, pads, dilations;
        std::int64_t storage_order;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MAX_POOLING_HPP */
