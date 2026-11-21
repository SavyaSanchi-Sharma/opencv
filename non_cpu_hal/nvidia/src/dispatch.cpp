// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "precomp.hpp"

#include "opencv2/imgproc.hpp"

namespace {

using namespace cv;

bool prepareDst(InputArray _src, OutputArray _dst, int rows, int cols, int type)
{
    if (!_dst.isUMat())
        return false;

    UMat& dst = _dst.getUMatRef();
    if (!dst.u && !dst.allocator)
        dst.allocator = cuda::getCudaAllocator();

    _dst.create(rows, cols, type);

    if (!cuda::isCudaUMat(_dst))
        return false;

    return _src.getUMat().u != _dst.getUMat().u;
}

bool finished()
{
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess)
        e = cudaStreamSynchronize(0);
    if (e != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}

}

bool cv::cuda_hal::tryFlip(InputArray _src, OutputArray _dst, int flipCode)
{
    if (flipCode != 0 || _src.dims() > 2 || _src.type() != CV_8UC1)
        return false;
    if (!cuda::isCudaUMat(_src))
        return false;

    const Size size = _src.size();
    if (size.width % 4 != 0)
        return false;
    if (!prepareDst(_src, _dst, size.height, size.width, CV_8UC1))
        return false;

    cv::cuda_hal::flip(_src, _dst, flipCode);
    return finished();
}

bool cv::cuda_hal::tryTranspose2d(InputArray _src, OutputArray _dst)
{
    if (_src.dims() > 2 || _src.type() != CV_8UC1)
        return false;
    if (!cuda::isCudaUMat(_src))
        return false;

    const Size size = _src.size();
    if (size.width == 0 || size.height == 0)
        return false;
    if (!prepareDst(_src, _dst, size.width, size.height, CV_8UC1))
        return false;

    cv::cuda_hal::transpose(_src, _dst);
    return finished();
}

bool cv::cuda_hal::tryThreshold(InputArray _src, OutputArray _dst,
                                double thresh, double maxval, int type)
{
    CV_UNUSED(maxval);

    if (type != THRESH_TRUNC || _src.dims() > 2 || _src.type() != CV_32FC1)
        return false;
    if (!cuda::isCudaUMat(_src))
        return false;

    const Size size = _src.size();
    if (size.width % 4 != 0)
        return false;
    if (!prepareDst(_src, _dst, size.height, size.width, CV_32FC1))
        return false;

    cv::cuda_hal::threshold(_src, _dst, thresh);
    return finished();
}

bool cv::cuda_hal::tryBoxFilter(InputArray _src, OutputArray _dst, int ddepth,
                                Size ksize, Point anchor, bool normalize, int borderType)
{
    if (!normalize || (borderType & ~BORDER_ISOLATED) != BORDER_REPLICATE)
        return false;
    if (_src.dims() > 2 || _src.type() != CV_32FC1)
        return false;
    if (ddepth >= 0 && ddepth != CV_32F)
        return false;
    if (ksize.width != ksize.height || ksize.width < 1 || ksize.width % 2 == 0)
        return false;
    if ((anchor.x != -1 && anchor.x != ksize.width / 2) ||
        (anchor.y != -1 && anchor.y != ksize.height / 2))
        return false;
    if (!cuda::isCudaUMat(_src))
        return false;

    const Size size = _src.size();
    if (!prepareDst(_src, _dst, size.height, size.width, CV_32FC1))
        return false;

    cv::cuda_hal::boxFilter(_src, _dst, ksize.width);
    return finished();
}
