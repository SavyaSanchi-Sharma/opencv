// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "common.hpp"

namespace cv { namespace cuda_hal { namespace device {

__global__ void transposeKernel(const void* src, size_t srcStep,
                                void* dst, size_t dstStep, int width, int height)
{
    __shared__ uchar tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    for (int j = 0; j < 32; j += 8) {
        if (x < width && (y + j) < height)
            tile[threadIdx.y + j][threadIdx.x] = rowPtr<uchar>(src, srcStep, y + j)[x];
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    for (int j = 0; j < 32; j += 8) {
        if (x < height && (y + j) < width)
            rowPtr<uchar>(dst, dstStep, y + j)[x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

void transpose(const void* src, size_t srcStep, void* dst, size_t dstStep,
               int width, int height, cudaStream_t stream)
{
    const dim3 blk(32, 8);
    const dim3 grid((width + 31) / 32, (height + 31) / 32);
    transposeKernel<<<grid, blk, 0, stream>>>(src, srcStep, dst, dstStep, width, height);
}

}}}
