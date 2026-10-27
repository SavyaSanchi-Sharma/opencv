// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_METAL_KERNELS_HPP
#define OPENCV_CORE_SRC_ACCEL_METAL_KERNELS_HPP

#include <stddef.h>
#include <stdint.h>

// Kernels of the Metal backend (core operations), as Metal Shading Language compiled at
// runtime with newLibraryWithSource:. Elements are moved as opaque groups of bytes, so one
// kernel serves every type of the same element size.
//
// Runtime compilation rather than a .metallib built ahead of time: it keeps the build to a
// single .mm file with no custom build step, the way the HIP backend would use hipRTC. The
// source below is a few hundred bytes and the compile happens once per device, behind a
// std::once_flag. A .metallib can be added later without changing the backend's shape.

namespace cv { namespace accel { namespace metal_impl { namespace kernels {

/* The same element sizes the HIP backend takes. Any size would in fact work here, because the
   byte copy is a loop over a function constant rather than a template, but a flip that
   succeeds on one backend and is declined by another for the same UMat would be a confusing
   thing to debug. Parity is worth more than the extra sizes. */
inline bool supportedElementSize(int esz)
{
    switch (esz)
    {
    case 1: case 2: case 3: case 4: case 6: case 8: case 12: case 16: case 24: case 32:
        return true;
    default:
        return false;
    }
}

/* Arguments of both kernels, bound at buffer index 2. The layout must match the MSL struct
   below, which it does field for field under both languages' natural alignment: six 8-byte
   members, then four 4-byte ones.

   The offsets are carried here rather than given to setBuffer:offset: on purpose. Metal wants
   a buffer offset to be a multiple of 4, and a UMat's offset is a multiple of its element
   size, which for CV_8UC3 is 3. Adding the offset inside the kernel has no such constraint. */
struct Params
{
    uint64_t soffset, doffset;      // bytes from the start of each buffer to the first element
    uint64_t sstep, sframe;         // bytes between rows, and between frames of a batch
    uint64_t dstep, dframe;
    uint32_t rows, cols;            // of the source
    int32_t flipRows, flipCols;     // flip only; transpose ignores them
};

//! the index each kernel argument is bound at, shared by the launcher and the MSL below
enum { ARG_SRC = 0, ARG_DST = 1, ARG_PARAMS = 2 };

//! the function constant holding the element size, specialized per pipeline
enum { FC_ELEMENT_SIZE = 0 };

inline const char* source()
{
    return R"MSL(
#include <metal_stdlib>
using namespace metal;

// Specialized when the pipeline is built, so the copy loop below unrolls into ESZ moves of a
// known count rather than a loop with a runtime bound.
constant int ESZ [[function_constant(0)]];

struct Params
{
    ulong soffset, doffset;
    ulong sstep, sframe;
    ulong dstep, dframe;
    uint rows, cols;
    int flipRows, flipCols;
};

static inline void copyElement(device const uchar* s, device uchar* d)
{
    for (int i = 0; i < ESZ; ++i)
        d[i] = s[i];
}

// OpenCV's flip codes, resolved to the two booleans by the host: 0 flips around the x axis
// (rows), > 0 around the y axis (columns), < 0 both.
kernel void accel_flip(device const uchar* src [[buffer(0)]],
                       device uchar* dst [[buffer(1)]],
                       constant Params& p [[buffer(2)]],
                       uint3 gid [[thread_position_in_grid]])
{
    // the grid is rounded up to whole threadgroups, so the edges have to be dropped here
    if (gid.x >= p.cols || gid.y >= p.rows)
        return;

    device const uchar* s = src + p.soffset + (ulong)gid.z * p.sframe;
    device uchar* d = dst + p.doffset + (ulong)gid.z * p.dframe;

    const uint sx = p.flipCols ? p.cols - 1 - gid.x : gid.x;
    const uint sy = p.flipRows ? p.rows - 1 - gid.y : gid.y;

    copyElement(s + (ulong)sy * p.sstep + (ulong)sx * ESZ,
                d + (ulong)gid.y * p.dstep + (ulong)gid.x * ESZ);
}

// rows x cols in, cols x rows out, per frame of the batch
kernel void accel_transpose(device const uchar* src [[buffer(0)]],
                            device uchar* dst [[buffer(1)]],
                            constant Params& p [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.cols || gid.y >= p.rows)
        return;

    device const uchar* s = src + p.soffset + (ulong)gid.z * p.sframe;
    device uchar* d = dst + p.doffset + (ulong)gid.z * p.dframe;

    copyElement(s + (ulong)gid.y * p.sstep + (ulong)gid.x * ESZ,
                d + (ulong)gid.x * p.dstep + (ulong)gid.y * ESZ);
}
)MSL";
}

}}}} // namespace cv::accel::metal_impl::kernels

#endif // OPENCV_CORE_SRC_ACCEL_METAL_KERNELS_HPP
