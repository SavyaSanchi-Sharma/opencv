// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHAPE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHAPE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/pointer.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <algorithm>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    class ShapeOp final : public CUDABackendNode {
    public:
        ShapeOp(csl::Stream stream_, int start_, int end_, DataLayout origLayout_)
            : stream(std::move(stream_)), start(start_), end(end_), origLayout(origLayout_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            MatShape inpShape = shape(inputs[0]);
            int outDims = inpShape.layout == DATA_LAYOUT_BLOCK ? inpShape.dims - 1 : inpShape.dims;
            int start_ = start < 0 ? start + outDims : start;
            int end_ = end >= outDims ? outDims : end < 0 ? end + outDims : end;
            start_ = std::max(0, std::min(start_, outDims));
            end_ = std::max(0, std::min(end_, outDims));

            std::int64_t shapeData[CV_MAX_DIM];
            if (inpShape.layout != DATA_LAYOUT_BLOCK)
            {
                for (int i = start_; i < end_; i++)
                    shapeData[i - start_] = (std::int64_t)inpShape[i];
            }
            else
            {
                const int semanticNdims = inpShape.dims - 1;
                std::vector<std::int64_t> semanticShape(semanticNdims);
                semanticShape[0] = (std::int64_t)inpShape[0];
                if (origLayout == DATA_LAYOUT_NCHW)
                {
                    semanticShape[1] = (std::int64_t)inpShape.C;
                    for (int i = 2; i < semanticNdims; i++)
                        semanticShape[i] = (std::int64_t)inpShape[i];
                }
                else
                {
                    semanticShape[semanticNdims - 1] = (std::int64_t)inpShape.C;
                    for (int i = 1; i < semanticNdims - 1; i++)
                        semanticShape[i] = (std::int64_t)inpShape[i + 1];
                }
                for (int i = start_; i < end_; i++)
                    shapeData[i - start_] = semanticShape[i];
            }

            auto output = csl::spanOf<std::int64_t>(outputs[0]);
            CV_Assert(output.size() == (std::size_t)(end_ - start_));
            csl::memcpy<std::int64_t>(output.get(), shapeData, output.size(), stream);
        }

    private:
        csl::Stream stream;
        int start, end;
        DataLayout origLayout;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHAPE_HPP */
