// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#ifndef OPENCV_IMGPROC_NON_CPU_HAL_REPLACEMENT_HPP
#define OPENCV_IMGPROC_NON_CPU_HAL_REPLACEMENT_HPP

#include "opencv2/core.hpp"

//! @cond IGNORED

/**
   @brief Threshold a device-resident array, leaving the result on the device.
   @param src source array
   @param dst destination array
   @param thresh threshold value
   @param maxval maximum value, used by the binary modes
   @param type threshold type, as passed to cv::threshold
   @return true when the backend handled the call, false to fall through to the next implementation
 */
inline bool nc_hal_ni_threshold(cv::InputArray src, cv::OutputArray dst,
                                double thresh, double maxval, int type)
{ CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(thresh); CV_UNUSED(maxval); CV_UNUSED(type); return false; }

#define cv_non_cpu_hal_threshold nc_hal_ni_threshold

/**
   @brief Box filter a device-resident array, leaving the result on the device.
   @param src source array
   @param dst destination array
   @param ddepth destination depth, negative to match the source
   @param ksize kernel size
   @param anchor kernel anchor, (-1,-1) for the centre
   @param normalize whether the kernel is normalized by its area
   @param border_type border extrapolation mode
   @return true when the backend handled the call, false to fall through to the next implementation
 */
inline bool nc_hal_ni_boxFilter(cv::InputArray src, cv::OutputArray dst, int ddepth,
                                cv::Size ksize, cv::Point anchor, bool normalize, int border_type)
{ CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(ddepth); CV_UNUSED(ksize); CV_UNUSED(anchor);
  CV_UNUSED(normalize); CV_UNUSED(border_type); return false; }

#define cv_non_cpu_hal_boxFilter nc_hal_ni_boxFilter

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
