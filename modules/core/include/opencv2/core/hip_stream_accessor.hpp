// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_HIP_STREAM_ACCESSOR_HPP
#define OPENCV_CORE_HIP_STREAM_ACCESSOR_HPP

#ifndef __cplusplus
#  error hip.hpp header must be compiled as C++
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "opencv2/core/hip.hpp"
namespace cv{
    namespace hip{


        struct StreamAccessor{
            
            CV_EXPORTS static hipStream_t getStream(const Stream& stream);
            CV_EXPORTS static Stream wrapStream(hipStream_t stream);
        };
        struct EventAccessor
        {
            CV_EXPORTS static hipEvent_t getEvent(const Event& event);
            CV_EXPORTS static Event wrapEvent(hipEvent_t event);
        };
        
    }
}
#endif /*for OPENCV_CORE_HIP_HPP*/