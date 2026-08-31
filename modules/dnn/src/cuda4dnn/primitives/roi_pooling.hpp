// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_POOLING_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/roi_pooling.hpp"

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ROIPoolingOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ROIPoolingOp(csl::Stream stream_, float spatial_scale)
            : stream(std::move(stream_)), spatial_scale{spatial_scale} { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto rois = csl::viewOf<T>(inputs[1]);
            auto output = csl::spanOf<T>(outputs[0]);

            kernels::roi_pooling<T>(stream, output, input, rois, spatial_scale);
        }

    private:
        csl::Stream stream;
        float spatial_scale;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_POOLING_HPP */
