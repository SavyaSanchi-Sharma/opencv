// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/stream.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/scale_shift.hpp"

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class SoftmaxOp final : public CUDABackendNode {
    public:
        SoftmaxOp(csl::cudnn::Handle handle, csl::Stream stream_, std::size_t axis_, bool log_,
                  float scale_ = 1.f)
            : cudnnHandle(std::move(handle)), stream(std::move(stream_)),
              channel_axis{ axis_ }, log{ log_ }, scale{ scale_ }
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == outputs.size());

            for (size_t i = 0; i < inputs.size(); i++)
            {
                auto input = csl::viewOf<T>(inputs[i]);
                auto output = csl::spanOf<T>(outputs[i]);

                if (scale != 1.f) {
                    // softmax(s*x) does not reduce to a rescale, so fold the scale in before max/exp
                    kernels::scale1_with_bias1<T>(stream, output, input,
                                                  static_cast<T>(scale), static_cast<T>(0.f));
                    input = csl::viewOf<T>(outputs[i]);
                }

                csl::tensor_ops::softmax<T>(cudnnHandle, output, input, channel_axis, log);
            }
        }

    private:
        csl::cudnn::Handle cudnnHandle;
        csl::Stream stream;
        std::size_t channel_axis;
        bool log;
        float scale;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_HPP */
