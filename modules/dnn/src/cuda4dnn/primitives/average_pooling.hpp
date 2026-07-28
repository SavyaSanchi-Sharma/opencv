// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_AVERAGE_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_AVERAGE_POOLING_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/average_pooling.hpp"

#ifdef HAVE_CUDNNJIT
#include "../csl/cudnn.hpp"
#include "../csl/cudnn/graph.hpp"
#include "../csl/workspace.hpp"
#include "../kernels/permute.hpp"
#endif

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct AveragePoolingConfiguration {
        std::vector<std::int64_t> kernel_shape;
        std::vector<std::int64_t> strides;
        std::vector<std::int64_t> pads;
        std::vector<std::int64_t> dilations;
        bool count_include_pad;
        bool ceil_mode = false;
    };

    template <class T>
    class AveragePoolingOp final : public CUDABackendNode {
    public:
        AveragePoolingOp(csl::Stream stream_,
#ifdef HAVE_CUDNNJIT
                         csl::cudnn::Handle cudnnHandle_,
#endif
                         const AveragePoolingConfiguration& config,
                         const MatShape& input_shape, const MatShape& output_shape)
            : stream(std::move(stream_)),
#ifdef HAVE_CUDNNJIT
              cudnnHandle(std::move(cudnnHandle_)),
#endif
              kernel_shape(config.kernel_shape),
              strides(config.strides),
              pads(config.pads),
              dilations(config.dilations),
              count_include_pad(config.count_include_pad)
        {
#ifdef HAVE_CUDNNJIT
            bool no_dilation = std::all_of(dilations.begin(), dilations.end(),
                                           [](std::int64_t d) { return d == 1; });
            if (!config.ceil_mode && no_dilation && kernel_shape.size() == 2 &&
                input_shape.dims == 4 && output_shape.dims == 4)
            {
                try {
                    typename csl::cudnn::ResampleGraph<T>::params_type params;
                    params.input_shape  = { input_shape[0], input_shape[1], input_shape[2], input_shape[3] };
                    params.output_shape = { output_shape[0], output_shape[1], output_shape[2], output_shape[3] };
                    params.window = { kernel_shape[0], kernel_shape[1] };
                    params.stride = { strides[0], strides[1] };
                    params.padding_pre  = { pads[0], pads[1] };
                    params.padding_post = { pads[2], pads[3] };
                    params.mode = count_include_pad ? CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING
                                                     : CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;

                    jitResampler = csl::cudnn::ResampleGraph<T>(cudnnHandle, params);
                    jitInputNHWC  = csl::Tensor<T>(input_shape[0], input_shape[2], input_shape[3], input_shape[1]);
                    jitOutputNHWC = csl::Tensor<T>(output_shape[0], output_shape[2], output_shape[3], output_shape[1]);
                    useJit = true;
                } catch (const cv::Exception&) {
                    useJit = false;
                }
            }
#else
            CV_UNUSED(input_shape);
            CV_UNUSED(output_shape);
#endif
        }

#ifdef HAVE_CUDNNJIT
        std::size_t get_workspace_memory_in_bytes() const noexcept override
        {
            return useJit ? jitResampler.get_workspace_size() : 0;
        }
#endif

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

#ifdef HAVE_CUDNNJIT
            if (useJit) {
                kernels::permute<T>(stream, jitInputNHWC, input, {0, 2, 3, 1});
                csl::WorkspaceAllocator allocator(workspace);
                auto scratch = allocator.get_instance();
                jitResampler.resample(cudnnHandle, jitInputNHWC.get(), jitOutputNHWC.get(), scratch);
                kernels::permute<T>(stream, output, jitOutputNHWC, {0, 3, 1, 2});
                return;
            }
#endif
            CV_UNUSED(workspace);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            MatShape outShape = cv::dnn::shape(outputs[0]);
            std::vector<std::int64_t> in_shape(inShape.begin(), inShape.end());
            std::vector<std::int64_t> out_shape(outShape.begin(), outShape.end());

            kernels::average_pool<T>(stream, output, input,
                                     in_shape, out_shape, kernel_shape, strides, pads, dilations, count_include_pad);
        }

    private:
        csl::Stream stream;
#ifdef HAVE_CUDNNJIT
        csl::cudnn::Handle cudnnHandle;
        bool useJit = false;
        csl::cudnn::ResampleGraph<T> jitResampler;
        csl::Tensor<T> jitInputNHWC, jitOutputNHWC;
#endif
        std::vector<std::int64_t> kernel_shape, strides, pads, dilations;
        bool count_include_pad;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_AVERAGE_POOLING_HPP */
