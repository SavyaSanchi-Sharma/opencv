// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_ACCEL_HIP_COMMON_HPP
#define OPENCV_HAL_ACCEL_HIP_COMMON_HPP

#include <hip/hip_runtime.h>

#include <limits>

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

template<int N> struct Bytes { unsigned char b[N]; };

const int BLOCK_X = 32, BLOCK_Y = 8;

// gridDim.y is limited: taller images are covered by a loop over y
const unsigned MAX_GRID_Y = 65535;

inline dim3 grid(int rows, int cols)
{
    const unsigned gy = (unsigned)((rows + BLOCK_Y - 1) / BLOCK_Y);
    return dim3((unsigned)((cols + BLOCK_X - 1) / BLOCK_X), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

#define CV_ACCEL_HIP_BY_SIZE(fn, ...) \
    switch (esz) \
    { \
    case 1: return fn<1>(__VA_ARGS__); \
    case 2: return fn<2>(__VA_ARGS__); \
    case 3: return fn<3>(__VA_ARGS__); \
    case 4: return fn<4>(__VA_ARGS__); \
    case 6: return fn<6>(__VA_ARGS__); \
    case 8: return fn<8>(__VA_ARGS__); \
    case 12: return fn<12>(__VA_ARGS__); \
    case 16: return fn<16>(__VA_ARGS__); \
    case 24: return fn<24>(__VA_ARGS__); \
    case 32: return fn<32>(__VA_ARGS__); \
    default: return hipErrorInvalidValue; \
    }

template<typename D, typename S>
__host__ __device__ __forceinline__ D hip_saturate_cast(S val) {
    const double d = static_cast<double>(val);
    if (d < static_cast<double>(std::numeric_limits<D>::lowest())) return std::numeric_limits<D>::lowest();
    if (d > static_cast<double>(std::numeric_limits<D>::max()))    return std::numeric_limits<D>::max();
    // Integer destinations round to nearest (matches saturate_cast/cvRound).
    return static_cast<D>(std::numeric_limits<D>::is_integer ? rint(d) : d);
}

#define HIP_SAT_FSPEC(D, S) \
    template<> __host__ __device__ __forceinline__ D hip_saturate_cast<D,S>(S v) { return static_cast<D>(v); }
HIP_SAT_FSPEC(float,  unsigned char)   HIP_SAT_FSPEC(float,  signed char)
HIP_SAT_FSPEC(float,  unsigned short)  HIP_SAT_FSPEC(float,  short)
HIP_SAT_FSPEC(float,  int)             HIP_SAT_FSPEC(float,  float)
HIP_SAT_FSPEC(float,  double)
HIP_SAT_FSPEC(double, unsigned char)   HIP_SAT_FSPEC(double, signed char)
HIP_SAT_FSPEC(double, unsigned short)  HIP_SAT_FSPEC(double, short)
HIP_SAT_FSPEC(double, int)             HIP_SAT_FSPEC(double, float)
HIP_SAT_FSPEC(double, double)
#undef HIP_SAT_FSPEC

template<typename T> struct HipVecTraits;

#define DEFINE_VT1(T_, ET_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return hip_saturate_cast<T_>(v[0]); \
        } \
    }
#define DEFINE_VT2(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return mk_(hip_saturate_cast<ET_>(v[0]), hip_saturate_cast<ET_>(v[1])); \
        } \
    }
#define DEFINE_VT3(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return mk_(hip_saturate_cast<ET_>(v[0]), hip_saturate_cast<ET_>(v[1]), \
                       hip_saturate_cast<ET_>(v[2])); \
        } \
    }
#define DEFINE_VT4(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return mk_(hip_saturate_cast<ET_>(v[0]), hip_saturate_cast<ET_>(v[1]), \
                       hip_saturate_cast<ET_>(v[2]), hip_saturate_cast<ET_>(v[3])); \
        } \
    }

DEFINE_VT1(unsigned char,  unsigned char);
DEFINE_VT2(uchar2,  unsigned char,  make_uchar2);
DEFINE_VT3(uchar3,  unsigned char,  make_uchar3);
DEFINE_VT4(uchar4,  unsigned char,  make_uchar4);
DEFINE_VT1(signed char,    signed char);
DEFINE_VT2(char2,   signed char,    make_char2);
DEFINE_VT3(char3,   signed char,    make_char3);
DEFINE_VT4(char4,   signed char,    make_char4);
DEFINE_VT1(unsigned short, unsigned short);
DEFINE_VT2(ushort2, unsigned short, make_ushort2);
DEFINE_VT3(ushort3, unsigned short, make_ushort3);
DEFINE_VT4(ushort4, unsigned short, make_ushort4);
DEFINE_VT1(short,   short);
DEFINE_VT2(short2,  short,  make_short2);
DEFINE_VT3(short3,  short,  make_short3);
DEFINE_VT4(short4,  short,  make_short4);
DEFINE_VT1(int,     int);
DEFINE_VT2(int2,    int,    make_int2);
DEFINE_VT3(int3,    int,    make_int3);
DEFINE_VT4(int4,    int,    make_int4);
DEFINE_VT1(float,   float);
DEFINE_VT2(float2,  float,  make_float2);
DEFINE_VT3(float3,  float,  make_float3);
DEFINE_VT4(float4,  float,  make_float4);
DEFINE_VT1(double,  double);
DEFINE_VT2(double2, double, make_double2);
DEFINE_VT3(double3, double, make_double3);
DEFINE_VT4(double4, double, make_double4);

#undef DEFINE_VT1
#undef DEFINE_VT2
#undef DEFINE_VT3
#undef DEFINE_VT4

template<typename A, typename B> struct HipLargerType                { typedef float  type; };
template<>                       struct HipLargerType<double, float> { typedef double type; };

}}}} // namespace cv::accel::hip_impl::kernels

#endif // OPENCV_HAL_ACCEL_HIP_COMMON_HPP
