// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_HIP_KERNELS_HPP
#define OPENCV_CORE_SRC_ACCEL_HIP_KERNELS_HPP

#include <hip/hip_runtime_api.h>
#include <stddef.h>

// Kernels of the HIP backend plugin (accel_kernels.hip), queued on @p s. Elements are @p esz
// bytes (supportedElementSize), rows are @p sstep / @p dstep bytes apart.
namespace cv { namespace accel { namespace hip_impl { namespace kernels {

bool supportedElementSize(int esz);

/* @p batch frames of @p rows x @p cols, frame f starting @p sframe (@p dframe) bytes into
   @p src (@p dst). A single image is batch 1 with the frame strides 0. */

//! OpenCV's flip codes: 0 around the x axis, > 0 around the y axis, < 0 both
hipError_t flip(const unsigned char* src, size_t sstep, size_t sframe,
                unsigned char* dst, size_t dstep, size_t dframe,
                int batch, int rows, int cols, int esz, int mode, hipStream_t s);
//! @p rows x @p cols in, cols x rows out, per frame
hipError_t transpose(const unsigned char* src, size_t sstep, size_t sframe,
                     unsigned char* dst, size_t dstep, size_t dframe,
                     int batch, int rows, int cols, int esz, hipStream_t s);

}}}} // namespace cv::accel::hip_impl::kernels

#endif // OPENCV_CORE_SRC_ACCEL_HIP_KERNELS_HPP
