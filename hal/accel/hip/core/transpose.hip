// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../common.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

namespace {

template<int N>
__global__ void transposeKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                                int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
        *(Bytes<N>*)(dst + (size_t)x * dstep + (size_t)y * N) =
            *(const Bytes<N>*)(src + (size_t)y * sstep + (size_t)x * N);
}

template<int N>
hipError_t transposeN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, hipStream_t s)
{
    hipLaunchKernelGGL(transposeKernel<N>, grid(rows, cols), dim3(BLOCK_X, BLOCK_Y), 0, s,
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
