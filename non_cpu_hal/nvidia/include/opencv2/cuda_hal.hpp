// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#ifndef OPENCV_NVIDIA_HAL_HPP
#define OPENCV_NVIDIA_HAL_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace cv { namespace cuda_hal {

CV_EXPORTS void flip(InputArray src, OutputArray dst, int flipCode, cuda::Stream& stream = cuda::Stream::Null());
CV_EXPORTS void lshift(InputArray src, OutputArray dst, int n, cuda::Stream& stream = cuda::Stream::Null());
CV_EXPORTS void threshold(InputArray src, OutputArray dst, double thresh, cuda::Stream& stream = cuda::Stream::Null());
CV_EXPORTS void transpose(InputArray src, OutputArray dst, cuda::Stream& stream = cuda::Stream::Null());
CV_EXPORTS void magnitude(InputArray src, OutputArray dst, cuda::Stream& stream = cuda::Stream::Null());
CV_EXPORTS void boxFilter(InputArray src, OutputArray dst, int ksize, cuda::Stream& stream = cuda::Stream::Null());

CV_EXPORTS bool tryFlip(InputArray src, OutputArray dst, int flipCode);
CV_EXPORTS bool tryTranspose2d(InputArray src, OutputArray dst);
CV_EXPORTS bool tryThreshold(InputArray src, OutputArray dst, double thresh, double maxval, int type);
CV_EXPORTS bool tryBoxFilter(InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor,
                             bool normalize, int borderType);

}}

#undef cv_non_cpu_hal_flip
#define cv_non_cpu_hal_flip cv::cuda_hal::tryFlip

#undef cv_non_cpu_hal_transpose2d
#define cv_non_cpu_hal_transpose2d cv::cuda_hal::tryTranspose2d

#undef cv_non_cpu_hal_threshold
#define cv_non_cpu_hal_threshold cv::cuda_hal::tryThreshold

#undef cv_non_cpu_hal_boxFilter
#define cv_non_cpu_hal_boxFilter cv::cuda_hal::tryBoxFilter

#endif
