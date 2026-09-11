// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#ifndef OPENCV_CORE_NON_CPU_HAL_REPLACEMENT_HPP
#define OPENCV_CORE_NON_CPU_HAL_REPLACEMENT_HPP

#include "opencv2/core.hpp"

//! @cond IGNORED

/**
   @brief Flip a device-resident array about the given axis, leaving the result on the device.
   @param src source array
   @param dst destination array
   @param flip_mode 0 flips around x-axis, positive around y-axis, negative both
   @return true when the backend handled the call, false to fall through to the next implementation
 */
inline bool nc_hal_ni_flip(cv::InputArray src, cv::OutputArray dst, int flip_mode)
{ CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(flip_mode); return false; }

#define cv_non_cpu_hal_flip nc_hal_ni_flip

/**
   @brief Transpose a device-resident 2D array, leaving the result on the device.
   @param src source array
   @param dst destination array
   @return true when the backend handled the call, false to fall through to the next implementation
 */
inline bool nc_hal_ni_transpose2d(cv::InputArray src, cv::OutputArray dst)
{ CV_UNUSED(src); CV_UNUSED(dst); return false; }

#define cv_non_cpu_hal_transpose2d nc_hal_ni_transpose2d

#include "custom_non_cpu_hal.hpp"

#define CV_NON_CPU_HAL_RUN_(condition, func, ...)                           \
try                                                                         \
{                                                                           \
    if ((condition) && (func))                                              \
    {                                                                       \
        CV_IMPL_ADD(CV_IMPL_CUDA);                                          \
        return __VA_ARGS__;                                                 \
    }                                                                       \
}                                                                           \
catch (const cv::Exception& e)                                              \
{                                                                           \
    CV_UNUSED(e);                                                           \
}

#define CV_NON_CPU_HAL_RUN(condition, func) CV_NON_CPU_HAL_RUN_(condition, func)

//! @endcond

#endif
