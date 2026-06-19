// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_HIPINL_HPP
#define OPENCV_CORE_HIPINL_HPP

#include "opencv2/core/hip.hpp"
namespace cv{
    namespace hip{

        //===================================================================================
        // Stream
        //===================================================================================

        inline
        Stream::Stream(const Ptr<Impl>& impl_)
            : impl(impl_)
        {
        }

        //===================================================================================
        // Event
        //===================================================================================

        inline
        Event::Event(const Ptr<Impl>& impl)
            : impl_(impl)
        {
        }
    }
}

#endif /*for OPENCV_CORE_HIPINL_HPP*/
