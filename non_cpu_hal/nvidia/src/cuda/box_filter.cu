// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "common.hpp"

namespace cv { namespace cuda_hal { namespace device {

__global__ void boxRowKernel(const void* src, size_t srcStep,
                             void* tmp, size_t tmpStep, int width, int height, int k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int r = k / 2;
    const float* srow = rowPtr<float>(src, srcStep, y);
    float s = 0.0f;
    for (int dx = -r; dx <= r; ++dx)
        s += srow[min(max(x + dx, 0), width - 1)];
    rowPtr<float>(tmp, tmpStep, y)[x] = s;
}

__global__ void boxColKernel(const void* tmp, size_t tmpStep,
                             void* dst, size_t dstStep, int width, int height, int k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int r = k / 2;
    float s = 0.0f;
    for (int dy = -r; dy <= r; ++dy)
        s += rowPtr<float>(tmp, tmpStep, min(max(y + dy, 0), height - 1))[x];
    rowPtr<float>(dst, dstStep, y)[x] = s / (float)(k * k);
}

void boxFilter(const void* src, size_t srcStep, void* tmp, size_t tmpStep,
               void* dst, size_t dstStep, int width, int height, int k, cudaStream_t stream)
{
    const dim3 blk(32, 8);
    const dim3 grid((width + 31) / 32, (height + 7) / 8);
    boxRowKernel<<<grid, blk, 0, stream>>>(src, srcStep, tmp, tmpStep, width, height, k);
    boxColKernel<<<grid, blk, 0, stream>>>(tmp, tmpStep, dst, dstStep, width, height, k);
}

}}}
