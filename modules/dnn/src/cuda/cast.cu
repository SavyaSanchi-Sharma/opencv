// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>

#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/cast.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        __global__ void cast_int64_to_fp32(const std::int64_t* input, float* output, std::size_t n) {
            for (auto i : grid_stride_range(n))
                output[i] = static_cast<float>(input[i]);
        }

        __global__ void cast_fp32_to_int64(const float* input, std::int64_t* output, std::size_t n) {
            for (auto i : grid_stride_range(n))
                output[i] = static_cast<std::int64_t>(input[i]);
        }

        template <class TOut, class TIn>
        struct Converter {
            __device__ static TOut apply(TIn v) { return static_cast<TOut>(v); }
        };

        template <class TIn>
        struct Converter<bool, TIn> {
            __device__ static bool apply(TIn v) { return v != TIn(0); }
        };

        template <>
        struct Converter<std::int32_t, std::int64_t> {
            __device__ static std::int32_t apply(std::int64_t v) {
                return v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : static_cast<std::int32_t>(v));
            }
        };

        template <class TOut, class TIn>
        __global__ void cast(Span<TOut> output, View<TIn> input) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = Converter<TOut, TIn>::apply(input[i]);
        }
    }

    template <class TOut, class TIn>
    void cast(const Stream& stream, Span<TOut> output, View<TIn> input) {
        CV_Assert(output.size() == input.size());
        if (output.size() == 0)
            return;
        auto kernel = raw::cast<TOut, TIn>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void cast(const Stream&, Span<bool>, View<bool>);
    template void cast(const Stream&, Span<bool>, View<std::int32_t>);
    template void cast(const Stream&, Span<bool>, View<std::int64_t>);
    template void cast(const Stream&, Span<bool>, View<float>);
    template void cast(const Stream&, Span<std::int32_t>, View<bool>);
    template void cast(const Stream&, Span<std::int32_t>, View<std::int32_t>);
    template void cast(const Stream&, Span<std::int32_t>, View<std::int64_t>);
    template void cast(const Stream&, Span<std::int32_t>, View<float>);
    template void cast(const Stream&, Span<std::int64_t>, View<bool>);
    template void cast(const Stream&, Span<std::int64_t>, View<std::int32_t>);
    template void cast(const Stream&, Span<std::int64_t>, View<std::int64_t>);
    template void cast(const Stream&, Span<std::int64_t>, View<float>);
    template void cast(const Stream&, Span<float>, View<bool>);
    template void cast(const Stream&, Span<float>, View<std::int32_t>);
    template void cast(const Stream&, Span<float>, View<std::int64_t>);
    template void cast(const Stream&, Span<float>, View<float>);

    void cast_int64_to_fp32(const Stream& stream, Span<float> output, View<std::int64_t> input) {
        CV_Assert(output.size() == input.size());
        auto kernel = raw::cast_int64_to_fp32;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, input.data().get(), output.data().get(), output.size());
    }

    void cast_fp32_to_int64(const Stream& stream, Span<std::int64_t> output, View<float> input) {
        CV_Assert(output.size() == input.size());
        auto kernel = raw::cast_fp32_to_int64;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, input.data().get(), output.data().get(), output.size());
    }

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
