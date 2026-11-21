// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "precomp.hpp"

namespace cv { namespace cuda_hal { namespace device {

void lshift(const void* src, size_t srcStep, void* dst, size_t dstStep,
            int width, int height, int n, cudaStream_t stream);
void thresholdTrunc(const void* src, size_t srcStep, void* dst, size_t dstStep,
                    int width, int height, float t, cudaStream_t stream);
void magnitude(const void* src, size_t srcStep, void* dst, size_t dstStep,
               int width, int height, cudaStream_t stream);

}}}

void cv::cuda_hal::lshift(InputArray _src, OutputArray _dst, int n, cuda::Stream& stream)
{
    CV_Assert(_src.type() == CV_8UC1);

    UMat src = _src.getUMat();
    CV_Assert(src.cols % 4 == 0);

    _dst.create(src.size(), src.type());
    UMat dst = _dst.getUMat();

    device::lshift(src.handle(ACCESS_READ), src.step,
                   dst.handle(ACCESS_WRITE), dst.step,
                   src.cols, src.rows, n, cuda::StreamAccessor::getStream(stream));
}

void cv::cuda_hal::threshold(InputArray _src, OutputArray _dst, double thresh, cuda::Stream& stream)
{
    CV_Assert(_src.type() == CV_32FC1);

    UMat src = _src.getUMat();
    CV_Assert(src.cols % 4 == 0);

    _dst.create(src.size(), src.type());
    UMat dst = _dst.getUMat();

    device::thresholdTrunc(src.handle(ACCESS_READ), src.step,
                           dst.handle(ACCESS_WRITE), dst.step,
                           src.cols, src.rows, static_cast<float>(thresh),
                           cuda::StreamAccessor::getStream(stream));
}

void cv::cuda_hal::magnitude(InputArray _src, OutputArray _dst, cuda::Stream& stream)
{
    CV_Assert(_src.type() == CV_32FC2);

    UMat src = _src.getUMat();
    CV_Assert(src.cols % 2 == 0);

    _dst.create(src.size(), CV_32FC1);
    UMat dst = _dst.getUMat();

    device::magnitude(src.handle(ACCESS_READ), src.step,
                      dst.handle(ACCESS_WRITE), dst.step,
                      src.cols, src.rows, cuda::StreamAccessor::getStream(stream));
}
