// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#ifndef OPENCV_NVIDIA_HAL_CUDA_COMMON_HPP
#define OPENCV_NVIDIA_HAL_CUDA_COMMON_HPP

#include <cuda_runtime.h>

#include "opencv2/core/hal/interface.h"

namespace cv { namespace cuda_hal { namespace device {

template <typename T>
__device__ __forceinline__ const T* rowPtr(const void* base, size_t step, int y)
{
    return reinterpret_cast<const T*>(static_cast<const char*>(base) + (size_t)y * step);
}

template <typename T>
__device__ __forceinline__ T* rowPtr(void* base, size_t step, int y)
{
    return reinterpret_cast<T*>(static_cast<char*>(base) + (size_t)y * step);
}

}}}

#endif
