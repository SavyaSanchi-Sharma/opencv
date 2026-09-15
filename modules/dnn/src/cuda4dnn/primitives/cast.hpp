// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CAST_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CAST_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../kernels/cast.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    // Only the (source, target) pairs this engine's ONNX graphs actually need
    // are implemented. Extend with more ops as new pairs are actually needed.
    class CastInt64ToFp32Op final : public CUDABackendNode {
    public:
        CastInt64ToFp32Op(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input = csl::viewOf<std::int64_t>(inputs[0]);
            auto output = csl::spanOf<float>(outputs[0]);
            kernels::cast_int64_to_fp32(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    class CastFp32ToInt64Op final : public CUDABackendNode {
    public:
        CastFp32ToInt64Op(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input = csl::viewOf<float>(inputs[0]);
            auto output = csl::spanOf<std::int64_t>(outputs[0]);
            kernels::cast_fp32_to_int64(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CAST_HPP */
