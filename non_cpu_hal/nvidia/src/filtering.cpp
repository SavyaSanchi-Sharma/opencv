// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "precomp.hpp"

namespace cv { namespace cuda_hal { namespace device {

void boxFilter(const void* src, size_t srcStep, void* tmp, size_t tmpStep,
               void* dst, size_t dstStep, int width, int height, int k, cudaStream_t stream);

}}}

namespace {

cv::UMat& scratch32f(int rows, int cols)
{
    thread_local cv::UMat* buf = new cv::UMat();
    buf->allocator = cv::cuda::getCudaAllocator();
    buf->create(rows, cols, CV_32FC1);
    return *buf;
}

}

void cv::cuda_hal::boxFilter(InputArray _src, OutputArray _dst, int ksize, cuda::Stream& stream)
{
    CV_Assert(_src.type() == CV_32FC1);
    CV_Assert(ksize > 0 && ksize % 2 == 1);

    UMat src = _src.getUMat();

    _dst.create(src.size(), src.type());
    UMat dst = _dst.getUMat();

    UMat& tmp = scratch32f(src.rows, src.cols);

    device::boxFilter(src.handle(ACCESS_READ), src.step,
                      tmp.handle(ACCESS_WRITE), tmp.step,
                      dst.handle(ACCESS_WRITE), dst.step,
                      src.cols, src.rows, ksize, cuda::StreamAccessor::getStream(stream));
}
