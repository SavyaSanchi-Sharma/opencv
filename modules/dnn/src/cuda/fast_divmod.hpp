// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA_FAST_DIVMOD_HPP
#define OPENCV_DNN_SRC_CUDA_FAST_DIVMOD_HPP

#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

template <typename T>
struct DivMod {
    DivMod(T d = 1) {
        d_ = d == 0 ? 1 : d;
        CV_Assert(d_ >= 1 && d_ <= std::numeric_limits<T>::max());
    }

    __host__ __device__ inline T div(T n) const {
        return n / d_;
    }

    __host__ __device__ inline T mod(T n) const {
        return n % d_;
    }

    __host__ __device__ inline void divmod(T n, T& q, T& r) const {
        q = div(n);
        r = n - q * d_;
    }

    T d_;
};

template <>
struct DivMod<int> {
    DivMod(int d = 1) {
        d_ = d == 0 ? 1 : d;
        CV_Assert(d_ >= 1 && d_ <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

        for (l_ = 0; l_ < 32; l_++)
            if ((1U << l_) >= d_) break;

        uint64_t one = 1;
        uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
        M_ = static_cast<uint32_t>(m);
        CV_Assert(M_ > 0 && M_ == m);
    }

    __host__ __device__ inline int div(int n) const {
#if defined(__CUDA_ARCH__)
        uint32_t t = __umulhi(M_, n);
        return (t + n) >> l_;
#else
        uint64_t t = ((uint64_t)M_ * n) >> 32;
        return static_cast<int>((t + n) >> l_);
#endif
    }

    __host__ __device__ inline int mod(int n) const {
        return n - div(n) * d_;
    }

    __host__ __device__ inline void divmod(int n, int& q, int& r) const {
        q = div(n);
        r = n - q * d_;
    }

    uint32_t d_;
    uint32_t M_;
    uint32_t l_;
};

using fast_divmod = DivMod<int>;

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_FAST_DIVMOD_HPP */
