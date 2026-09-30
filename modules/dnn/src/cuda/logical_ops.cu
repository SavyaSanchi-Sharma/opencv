// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "kernel_dispatcher.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

#include "../cuda4dnn/kernels/logical_ops.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        /* maps an output index to an offset in each of N broadcast inputs; a zero stride marks a broadcast axis */
        template <std::size_t Rank, std::size_t N>
        struct BroadcastIndex {
            array<size_type, Rank> out_strides;
            array<size_type, Rank> in_strides[N];

            __device__ void input_offsets(index_type i, index_type (&offsets)[N]) const {
                for (int k = 0; k < N; k++)
                    offsets[k] = 0;
                index_type remaining = i;
                for (int axis = 0; axis < Rank; axis++) {
                    index_type coord = remaining;
                    if (axis != Rank - 1) {
                        coord = remaining / out_strides[axis];
                        remaining -= coord * out_strides[axis];
                    }
                    for (int k = 0; k < N; k++)
                        offsets[k] += coord * in_strides[k][axis];
                }
            }
        };

        struct EqualFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a == b; }
        };

        struct GreaterFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a > b; }
        };

        struct GreaterEqualFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a >= b; }
        };

        struct LessFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a < b; }
        };

        struct LessEqualFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a <= b; }
        };

        struct AndFunctor {
            __device__ bool operator()(bool a, bool b) const { return a && b; }
        };

        struct OrFunctor {
            __device__ bool operator()(bool a, bool b) const { return a || b; }
        };

        struct XorFunctor {
            __device__ bool operator()(bool a, bool b) const { return a != b; }
        };

        template <class T, class Functor, std::size_t Rank>
        __global__ void binary_to_bool(Span<bool> output, View<T> x, View<T> y, BroadcastIndex<Rank, 2> index)
        {
            Functor functor;
            for (auto i : grid_stride_range(output.size())) {
                index_type offsets[2];
                index.input_offsets(i, offsets);
                output[i] = functor(x[offsets[0]], y[offsets[1]]);
            }
        }

        template <class T, std::size_t Rank>
        __global__ void where(Span<T> output, View<bool> cond, View<T> x, View<T> y, BroadcastIndex<Rank, 3> index)
        {
            for (auto i : grid_stride_range(output.size())) {
                index_type offsets[3];
                index.input_offsets(i, offsets);
                output[i] = cond[offsets[0]] ? x[offsets[1]] : y[offsets[2]];
            }
        }

        __global__ void logical_not(Span<bool> output, View<bool> input)
        {
            for (auto id : grid_stride_range(output.size()))
                output[id] = !input[id];
        }
    }

    /* Left-pads every input shape to the output rank and returns that rank. When all shapes
     * match there is no broadcasting, so every shape collapses to a single axis. */
    template <class Shape> static
    int normalize_broadcast_shapes(Shape& out_shape, std::vector<Shape>& in_shapes)
    {
        CV_Assert(in_shapes.size() <= 3);

        if (out_shape.empty())
            out_shape.assign(1, 1);
        bool same = true;
        for (Shape& shape : in_shapes) {
            if (shape.empty())
                shape.assign(1, 1);
            same = same && shape == out_shape;
        }
        if (same) {
            std::size_t total = 1;
            for (auto dim : out_shape)
                total *= dim;
            out_shape.assign(1, total);
            for (Shape& shape : in_shapes)
                shape = out_shape;
        }

        const int rank = static_cast<int>(out_shape.size());
        CV_Assert(rank >= 1 && rank <= CSL_MAX_TENSOR_RANK);

        for (Shape& shape : in_shapes) {
            CV_Assert(shape.size() <= out_shape.size());
            shape.insert(shape.begin(), out_shape.size() - shape.size(), 1);
            for (int axis = 0; axis < rank; axis++)
                CV_Assert(shape[axis] == 1 || shape[axis] == out_shape[axis]);
        }
        return rank;
    }

    template <std::size_t Rank, std::size_t N, class Shape> static
    raw::BroadcastIndex<Rank, N> make_broadcast_index(const Shape& out_shape, const std::vector<Shape>& in_shapes)
    {
        CV_Assert(out_shape.size() == Rank && in_shapes.size() == N);

        raw::BroadcastIndex<Rank, N> index;
        size_type stride = 1;
        for (int axis = Rank - 1; axis >= 0; --axis) {
            index.out_strides[axis] = stride;
            stride *= static_cast<size_type>(out_shape[axis]);
        }
        for (std::size_t k = 0; k < N; k++) {
            size_type in_stride = 1;
            for (int axis = Rank - 1; axis >= 0; --axis) {
                const auto dim = static_cast<size_type>(in_shapes[k][axis]);
                index.in_strides[k][axis] = dim == 1 ? 0 : in_stride;
                in_stride *= dim;
            }
        }
        return index;
    }

    /* the CSL index types are 32-bit, like the rest of the CUDA backend */
    static void check_index_range(std::size_t size) {
        CV_Assert(size <= static_cast<std::size_t>(std::numeric_limits<index_type>::max()));
    }

    template <class T, class Functor, std::size_t Rank, class Shape> static
    void launch_binary_to_bool_kernel(const Stream& stream, Span<bool> output, View<T> x, View<T> y,
                                      const Shape& out_shape, const std::vector<Shape>& in_shapes)
    {
        auto kernel = raw::binary_to_bool<T, Functor, Rank>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, x, y, make_broadcast_index<Rank, 2>(out_shape, in_shapes));
    }

    GENERATE_KERNEL_DISPATCHER_2TP(binary_to_bool_dispatcher, launch_binary_to_bool_kernel);

    template <class T, class Functor> static
    void launch_binary_to_bool(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y)
    {
        if (output.size() == 0)
            return;
        check_index_range(output.size());

        auto out_shape = output.shape_as_vector();
        std::vector<decltype(out_shape)> in_shapes{x.shape_as_vector(), y.shape_as_vector()};
        const int rank = normalize_broadcast_shapes(out_shape, in_shapes);
        binary_to_bool_dispatcher<T, Functor, 1, CSL_MAX_TENSOR_RANK>(rank, stream, Span<bool>(output), View<T>(x), View<T>(y),
                                                                       out_shape, in_shapes);
    }

    template <class T>
    void compare_equal(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::EqualFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_greater(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::GreaterFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_greater_equal(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::GreaterEqualFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_less(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::LessFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_less_equal(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::LessEqualFunctor>(stream, output, x, y);
    }

    void logical_and(const Stream& stream, TensorSpan<bool> output, TensorView<bool> x, TensorView<bool> y) {
        launch_binary_to_bool<bool, raw::AndFunctor>(stream, output, x, y);
    }

    void logical_or(const Stream& stream, TensorSpan<bool> output, TensorView<bool> x, TensorView<bool> y) {
        launch_binary_to_bool<bool, raw::OrFunctor>(stream, output, x, y);
    }

    void logical_xor(const Stream& stream, TensorSpan<bool> output, TensorView<bool> x, TensorView<bool> y) {
        launch_binary_to_bool<bool, raw::XorFunctor>(stream, output, x, y);
    }

    void logical_not(const Stream& stream, TensorSpan<bool> output, TensorView<bool> input) {
        CV_Assert(output.size() == input.size());
        if (output.size() == 0)
            return;

        auto kernel = raw::logical_not;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, Span<bool>(output), View<bool>(input));
    }

    template <class T, std::size_t Rank, class Shape> static
    void launch_where_kernel(const Stream& stream, Span<T> output, View<bool> cond, View<T> x, View<T> y,
                             const Shape& out_shape, const std::vector<Shape>& in_shapes)
    {
        auto kernel = raw::where<T, Rank>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, cond, x, y, make_broadcast_index<Rank, 3>(out_shape, in_shapes));
    }

    GENERATE_KERNEL_DISPATCHER(where_dispatcher, launch_where_kernel);

    template <class T>
    void where(const Stream& stream, TensorSpan<T> output, TensorView<bool> cond, TensorView<T> x, TensorView<T> y) {
        if (output.size() == 0)
            return;
        check_index_range(output.size());

        auto out_shape = output.shape_as_vector();
        std::vector<decltype(out_shape)> in_shapes{cond.shape_as_vector(), x.shape_as_vector(), y.shape_as_vector()};
        const int rank = normalize_broadcast_shapes(out_shape, in_shapes);
        where_dispatcher<T, 1, CSL_MAX_TENSOR_RANK>(rank, stream, Span<T>(output), View<bool>(cond), View<T>(x), View<T>(y),
                                                    out_shape, in_shapes);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void where(const Stream&, TensorSpan<__half>, TensorView<bool>, TensorView<__half>, TensorView<__half>);
#endif
    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void where(const Stream&, TensorSpan<float>, TensorView<bool>, TensorView<float>, TensorView<float>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void where(const Stream&, TensorSpan<int8_t>, TensorView<bool>, TensorView<int8_t>, TensorView<int8_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void where(const Stream&, TensorSpan<uint8_t>, TensorView<bool>, TensorView<uint8_t>, TensorView<uint8_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void where(const Stream&, TensorSpan<int32_t>, TensorView<bool>, TensorView<int32_t>, TensorView<int32_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void where(const Stream&, TensorSpan<int64_t>, TensorView<bool>, TensorView<int64_t>, TensorView<int64_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void where(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>, TensorView<bool>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
