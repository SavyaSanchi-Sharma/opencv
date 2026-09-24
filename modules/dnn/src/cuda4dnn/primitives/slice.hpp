// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SLICE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SLICE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/slice.hpp"
#include "../kernels/fill_copy.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <functional>
#include <vector>
#include <algorithm>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    using SliceOffsetResolver = std::function<void(const std::vector<UMat>&, std::vector<std::size_t>&,
                                                   std::vector<std::size_t>&)>;

    template <class T>
    class SliceOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        /* offsets/steps are indexed by output number and each subvector is indexed by axis number */
        SliceOp(csl::Stream stream_, std::vector<std::vector<std::size_t>> offsets,
                std::vector<std::vector<std::size_t>> steps = {})
            : stream(std::move(stream_)), offsets(std::move(offsets)), steps(std::move(steps))
        {
        }

        SliceOp(csl::Stream stream_, SliceOffsetResolver resolver_)
            : stream(std::move(stream_)), offsets(1), steps(1), resolver(std::move(resolver_))
        {
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            /* sometimes the output shape is passed in the form of a second input tensor
             * it's only required for initialization and not here
             */
            CV_Assert(inputs.size() >= 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            CV_Assert(offsets.size() == outputs.size());

            for (int i = 0; i < outputs.size(); ++i)
            {
                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                kernels::slice<T>(stream, output, input, offsets[i], (size_t)i < steps.size() ? steps[i] : std::vector<std::size_t>());
            }
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 1);

            if (resolver)
                resolver(inputs, offsets[0], steps[0]);

            auto input = csl::viewOf<T>(inputs[0]);

            CV_Assert(offsets.size() == outputs.size());
            const MatShape inShape = cv::dnn::shape(inputs[0]);
            for (int i = 0; i < (int)outputs.size(); ++i)
            {
                const MatShape outShape = cv::dnn::shape(outputs[i]);
                CV_Assert(outShape.dims == inShape.dims && offsets[i].size() == (std::size_t)inShape.dims);
                for (int a = 0; a < inShape.dims; a++) {
                    const std::int64_t step = (std::size_t)i < steps.size() && !steps[i].empty()
                                              ? static_cast<std::int64_t>(steps[i][a]) : 1;
                    const std::int64_t first = static_cast<std::int64_t>(offsets[i][a]);
                    const std::int64_t last = first + static_cast<std::int64_t>(outShape[a] - 1) * step;
                    if (outShape[a] > 0)
                        CV_Assert(0 <= first && first < inShape[a] && 0 <= last && last < inShape[a]);
                }
                auto output = csl::spanOf<T>(outputs[i]);
                kernels::slice<T>(stream, output, input, offsets[i], (size_t)i < steps.size() ? steps[i] : std::vector<std::size_t>());
            }
        }

    private:
        csl::Stream stream;
        std::vector<std::vector<std::size_t>> offsets;
        std::vector<std::vector<std::size_t>> steps;
        SliceOffsetResolver resolver;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SLICE_HPP */
