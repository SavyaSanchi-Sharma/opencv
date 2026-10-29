// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../common.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

namespace {

const int TILE = 32, TILE_ROWS = 8;

template<int N>
__global__ void transposeKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                                int rows, int cols)
{
    // +1 column so columns of the tile land in different LDS banks
    __shared__ Bytes<N> tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += TILE_ROWS)
        if (x < cols && y + j < rows)
            hipCopyN<N>(tile[threadIdx.y + j][threadIdx.x].b,
                        src + (size_t)(y + j) * sstep + (size_t)x * N);

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += TILE_ROWS)
        if (x < rows && y + j < cols)
            hipCopyN<N>(dst + (size_t)(y + j) * dstep + (size_t)x * N,
                        tile[threadIdx.x][threadIdx.y + j].b);
}

template<int N>
hipError_t transposeN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, hipStream_t s)
{
    const dim3 grid((unsigned)((cols + TILE - 1) / TILE), (unsigned)((rows + TILE - 1) / TILE));
    hipLaunchKernelGGL(transposeKernel<N>, grid, dim3(TILE, TILE_ROWS), 0, s,
                       src, sstep, dst, dstep, rows, cols);
    return hipGetLastError();
}

} // namespace

hipError_t transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                     int rows, int cols, int esz, hipStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return hipSuccess;
    CV_ACCEL_HIP_BY_SIZE(transposeN, src, sstep, dst, dstep, rows, cols, s)
}

}}}} // namespace cv::accel::hip_impl::kernels
