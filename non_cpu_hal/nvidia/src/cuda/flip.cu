// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "common.hpp"

namespace cv { namespace cuda_hal { namespace device {

__global__ void flipRowsKernel(const void* src, size_t srcStep,
                               void* dst, size_t dstStep, int width4, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width4 || y >= height) return;

    rowPtr<uchar4>(dst, dstStep, y)[x] = rowPtr<uchar4>(src, srcStep, height - 1 - y)[x];
}

void flipRows(const void* src, size_t srcStep, void* dst, size_t dstStep,
              int width, int height, cudaStream_t stream)
{
    const dim3 blk(32, 8);
    const dim3 grid((width / 4 + 31) / 32, (height + 7) / 8);
    flipRowsKernel<<<grid, blk, 0, stream>>>(src, srcStep, dst, dstStep, width / 4, height);
}

}}}
