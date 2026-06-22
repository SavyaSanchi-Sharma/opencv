// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_HIP_HPP
#define OPENCV_CORE_HIP_HPP

#ifndef __cplusplus
#  error hip.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/hip_types.hpp"

namespace cv{
    namespace hip{

    CV_EXPORTS_W bool useHip();
    CV_EXPORTS MatAllocator* getHipAllocator();

    //! Returns true if @p a is a UMat currently residing on a HIP device.
    //! Use this instead of repeating the currAllocator == getHipAllocator() check.
    CV_EXPORTS bool isHipUMat(InputArray a);

    //! @brief AOT-compiled HIP kernel launchers.
    //!
    //! These take the raw device handle + matrix metadata (pointer, row step,
    //! rows, cols, type) exactly as carried by a HIP-backed UMat (UMatData::handle
    //! and the UMat header).  This mirrors how the OpenCL backend feeds cl_mem +
    //! step into a kernel via ocl::KernelArg — there is no intermediate matrix
    //! type.  The UMat T-API dispatch points (UMat::setTo/copyTo/convertTo and the
    //! arithmetic ops) call these directly.
    namespace device {
        CV_EXPORTS void setToWithoutMask(void* data, size_t step, int rows, int cols, int type,
                                         Scalar val);
        CV_EXPORTS void setToWithMask(void* data, size_t step, int rows, int cols, int type,
                                      const void* mask, size_t maskStep,
                                      Scalar val);
        CV_EXPORTS void copyToWithMask(const void* src, size_t srcStep,
                                       void* dst, size_t dstStep,
                                       const void* mask, size_t maskStep,
                                       int rows, int cols, int type, int maskCn);
        CV_EXPORTS void convertToScale(const void* src, size_t srcStep, int stype,
                                       void* dst, size_t dstStep, int dtype,
                                       int rows, int cols, double alpha, double beta);
    } // namespace device


    //! @brief Returns the number of installed HIP-enabled devices.
    //! Returns 0 if HIP support is not compiled in or no device is found.
    CV_EXPORTS_W int getHipEnabledDeviceCount();

    //! @brief Sets the current HIP device.
    CV_EXPORTS_W void setDevice(int device);

    //! @brief Returns the current HIP device index.
    CV_EXPORTS_W int getDevice();

    //! @brief Explicitly destroys and cleans up all resources associated with
    //! the current device in the current process.
    CV_EXPORTS_W void resetDevice();

    enum FeatureSet
    {
        GLOBAL_ATOMICS,
        SHARED_ATOMICS,
        NATIVE_DOUBLE,
        WARP_SHUFFLE_FUNCTIONS,
        DYNAMIC_PARALLELISM
    };

    CV_EXPORTS bool deviceSupports(FeatureSet feature_set);

    class CV_EXPORTS_W TargetArchs
    {
    public:
        static bool builtWith(FeatureSet feature_set);

        CV_WRAP static bool has(int major, int minor);
        CV_WRAP static bool hasBin(int major, int minor);

        CV_WRAP static bool hasEqualOrGreater(int major, int minor);
        CV_WRAP static bool hasEqualOrGreaterBin(int major, int minor);
    };

    class CV_EXPORTS_W DeviceInfo
    {
    public:
        CV_WRAP DeviceInfo();
        CV_WRAP DeviceInfo(int device_id);

        CV_WRAP int deviceID() const;

        const char* name() const;

        CV_WRAP size_t totalGlobalMem() const;
        CV_WRAP size_t sharedMemPerBlock() const;
        CV_WRAP int regsPerBlock() const;
        CV_WRAP int warpSize() const;
        CV_WRAP size_t memPitch() const;
        CV_WRAP int maxThreadsPerBlock() const;
        CV_WRAP Vec3i maxThreadsDim() const;
        CV_WRAP Vec3i maxGridSize() const;
        CV_WRAP int clockRate() const;
        CV_WRAP size_t totalConstMem() const;
        CV_WRAP int majorVersion() const;
        CV_WRAP int minorVersion() const;
        CV_WRAP size_t textureAlignment() const;
        CV_WRAP size_t texturePitchAlignment() const;
        CV_WRAP int multiProcessorCount() const;
        CV_WRAP bool kernelExecTimeoutEnabled() const;
        CV_WRAP bool integrated() const;
        CV_WRAP bool canMapHostMemory() const;

        enum ComputeMode
        {
            ComputeModeDefault,
            ComputeModeExclusive,
            ComputeModeProhibited,
            ComputeModeExclusiveProcess
        };

        CV_WRAP DeviceInfo::ComputeMode computeMode() const;

        CV_WRAP int maxTexture1D() const;
        CV_WRAP int maxTexture1DLinear() const;
        CV_WRAP Vec2i maxTexture2D() const;
        CV_WRAP Vec3i maxTexture3D() const;

        CV_WRAP bool concurrentKernels() const;
        CV_WRAP bool ECCEnabled() const;

        CV_WRAP int pciBusID() const;
        CV_WRAP int pciDeviceID() const;
        CV_WRAP int pciDomainID() const;

        CV_WRAP bool tccDriver() const;

        CV_WRAP int memoryClockRate() const;
        CV_WRAP int memoryBusWidth() const;
        CV_WRAP int l2CacheSize() const;
        CV_WRAP int maxThreadsPerMultiProcessor() const;

        CV_WRAP void queryMemory(size_t& totalMemory, size_t& freeMemory) const;
        CV_WRAP size_t freeMemory() const;
        CV_WRAP size_t totalMemory() const;

        CV_WRAP const char* gcnArchName() const;
        CV_WRAP bool cooperativeLaunch() const;
        CV_WRAP bool isLargeBar() const;
        CV_WRAP int asicRevision() const;

        bool supports(FeatureSet feature_set) const;
        CV_WRAP bool isCompatible() const;

    private:
        int device_id_;
    };

    CV_EXPORTS_W void printHipDeviceInfo(int device);
    CV_EXPORTS_W void printShortHipDeviceInfo(int device);

    }
}

#endif /*for OPENCV_CORE_HIP_HPP*/

