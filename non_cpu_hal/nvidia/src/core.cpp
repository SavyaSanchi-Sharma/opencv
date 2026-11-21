// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "precomp.hpp"

namespace cv { namespace cuda_hal { namespace device {

void flipRows(const void* src, size_t srcStep, void* dst, size_t dstStep,
              int width, int height, cudaStream_t stream);
void transpose(const void* src, size_t srcStep, void* dst, size_t dstStep,
               int width, int height, cudaStream_t stream);

}}}

void cv::cuda_hal::flip(InputArray _src, OutputArray _dst, int flipCode, cuda::Stream& stream)
{
    CV_Assert(_src.type() == CV_8UC1);
    CV_Assert(flipCode == 0);

    UMat src = _src.getUMat();
    CV_Assert(src.cols % 4 == 0);

    _dst.create(src.size(), src.type());
    UMat dst = _dst.getUMat();

    device::flipRows(src.handle(ACCESS_READ), src.step,
                     dst.handle(ACCESS_WRITE), dst.step,
                     src.cols, src.rows, cuda::StreamAccessor::getStream(stream));
}

void cv::cuda_hal::transpose(InputArray _src, OutputArray _dst, cuda::Stream& stream)
{
    CV_Assert(_src.type() == CV_8UC1);

    UMat src = _src.getUMat();

    _dst.create(src.cols, src.rows, src.type());
    UMat dst = _dst.getUMat();

    device::transpose(src.handle(ACCESS_READ), src.step,
                      dst.handle(ACCESS_WRITE), dst.step,
                      src.cols, src.rows, cuda::StreamAccessor::getStream(stream));
}
