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
        class Stream;  // forward-declared here; defined later in this file

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
                                         Scalar val, Stream& stream);
        CV_EXPORTS void setToWithMask(void* data, size_t step, int rows, int cols, int type,
                                      const void* mask, size_t maskStep,
                                      Scalar val, Stream& stream);
        CV_EXPORTS void copyToWithMask(const void* src, size_t srcStep,
                                       void* dst, size_t dstStep,
                                       const void* mask, size_t maskStep,
                                       int rows, int cols, int type, int maskCn, Stream& stream);
        CV_EXPORTS void convertToNoScale(const void* src, size_t srcStep, int stype,
                                         void* dst, size_t dstStep, int dtype,
                                         int rows, int cols, Stream& stream);
        CV_EXPORTS void convertToScale(const void* src, size_t srcStep, int stype,
                                       void* dst, size_t dstStep, int dtype,
                                       int rows, int cols, double alpha, double beta, Stream& stream);
        CV_EXPORTS void multiplyF32(const void* src1, size_t step1,
                                    const void* src2, size_t step2,
                                    void* dst, size_t stepd,
                                    int rows, int cols, Stream& stream);
    } // namespace device


    class CV_EXPORTS_W Stream{
        typedef void(Stream::*bool_type)() const;
        void this_type_does_not_support_comparisions() const {}
        public:
            typedef void(*StreamCallback)(int status,void* userData);
            CV_WRAP Stream();
            CV_WRAP Stream(const size_t hipFlags);
            CV_WRAP bool queryIfComplete() const;
            CV_WRAP void waitForCompletion();
            CV_WRAP void enqueueHostCallback(StreamCallback callback, void* userData);
            CV_WRAP static Stream& Null();
            operator bool_type() const;
            CV_WRAP void* hipPtr() const;
            class Impl;
            private:
                Ptr<Impl>impl;
                Stream(const Ptr<Impl>& impl_) : impl(impl_) {}
                friend struct StreamAccessor;
                friend class DefaultDeviceInitializer;
            

    };

    CV_EXPORTS_W Stream wrapStream(size_t cudaStreamMemoryAddress);



    class CV_EXPORTS_W Event{
        public:
        enum CreateFlags{
            DEFAULT        = 0x00,  /**< Default event flag */
            BLOCKING_SYNC  = 0x01,  /**< Event uses blocking synchronization */
            DISABLE_TIMING = 0x02,  /**< Event will not record timing data */
            INTERPROCESS   = 0x04   /**< Event is suitable for interprocess use. DisableTiming must be set */
        };


        CV_WRAP explicit Event(const Event::CreateFlags flags = Event::CreateFlags::DEFAULT);

        //! records an event
        CV_WRAP void record(Stream& stream = Stream::Null());

        //! queries an event's status
        CV_WRAP bool queryIfComplete() const;

        //! waits for an event to complete
        CV_WRAP void waitForCompletion();

        //! computes the elapsed time between events
        CV_WRAP static float elapsedTime(const Event& start, const Event& end);

        class Impl;

    private:
        Ptr<Impl> impl_;
        Event(const Ptr<Impl>& impl) : impl_(impl) {}

        friend struct EventAccessor;
    };

    CV_ENUM_FLAGS(Event::CreateFlags)


    CV_EXPORTS_W int getHipEnabledDeviceCount();

    /** @brief Sets a device and initializes it for the current thread.

    @param device System index of a CUDA device starting with 0.

    If the call of this function is omitted, a default device is initialized at the fist CUDA usage.
    */
    CV_EXPORTS_W void setDevice(int device);

    /** @brief Returns the current device index set by cuda::setDevice or initialized by default.
     */
    CV_EXPORTS_W int getDevice();

    /** @brief Explicitly destroys and cleans up all resources associated with the current device in the current
    process.

    Any subsequent API call to this device will reinitialize the device.
    */
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

