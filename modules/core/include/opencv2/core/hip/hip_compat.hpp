#ifndef OPENCV_HIP_HIP_COMPAT_HPP
#define OPENCV_HIP_HIP_COMPAT_HPP

#include <hip/hip_runtime.h>
namespace cv {
    namespace hip{
        namespace device{
            namespace compat{
                using ulonglong4 = ::ulonglong4;
                using double4 = ::double4;
                __host__ __device__ __forceinline__
                double4 make_double4(double x, double y, double z, double w)
                {
                    return ::make_double4(x, y, z, w);
                }
            }
        }
    }
}