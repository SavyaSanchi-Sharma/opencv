// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "common.hpp"

namespace cv { namespace cuda_hal { namespace device {

__global__ void magnitudeKernel(const void* src, size_t srcStep,
                                void* dst, size_t dstStep, int width2, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width2 || y >= height) return;

    const float4 v = rowPtr<float4>(src, srcStep, y)[x];
    rowPtr<float2>(dst, dstStep, y)[x] = make_float2(sqrtf(v.x * v.x + v.y * v.y),
                                                     sqrtf(v.z * v.z + v.w * v.w));
}

void magnitude(const void* src, size_t srcStep, void* dst, size_t dstStep,
               int width, int height, cudaStream_t stream)
{
    const dim3 blk(32, 8);
    const dim3 grid((width / 2 + 31) / 32, (height + 7) / 8);
    magnitudeKernel<<<grid, blk, 0, stream>>>(src, srcStep, dst, dstStep, width / 2, height);
}

}}}
