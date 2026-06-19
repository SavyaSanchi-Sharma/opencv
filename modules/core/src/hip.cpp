// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#define OPENCV_CORE_HIP_IMPL
#include "precomp.hpp"
#include "opencv2/core/hip.hpp"
#include "opencv2/core/hip_stream_accessor.hpp"
#include "opencv2/core/private/hip_stubs.hpp"
#include "umatrix.hpp"

#ifdef HAVE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "opencv2/core/hipdev.hpp"
#endif

using namespace cv;
using namespace cv::hip;

#ifdef HAVE_HIP

#define hipSafeCall(expr) CV_HIP_SAFE_CALL(expr)

#endif

// ======================== HipAllocator ========================

#ifdef HAVE_HIP
namespace {

class HipAllocator CV_FINAL : public MatAllocator
{
public:
    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step,
                       AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        CV_UNUSED(flags); CV_UNUSED(usageFlags);
        CV_Assert(data == nullptr);

        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--) {
            if (step) step[i] = total;
            total *= sizes[i];
        }

        void* devicePtr = nullptr;
        hipSafeCall(hipMalloc(&devicePtr, total));

        UMatData* u = new UMatData(this);
        u->data     = nullptr;
        u->origdata = nullptr;
        u->handle   = devicePtr;
        u->size     = total;
        u->flags    = UMatData::COPY_ON_MAP;
        u->markHostCopyObsolete(true);
        return u;
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        CV_UNUSED(usageFlags); CV_UNUSED(accessFlags);
        if (!u) return false;

        UMatDataAutoLock lock(u);

        if (u->handle == nullptr) {
            CV_Assert(u->origdata != nullptr);
            void* devicePtr = nullptr;
            hipSafeCall(hipMalloc(&devicePtr, u->size));
            u->handle = devicePtr;
            if (u->origdata) {
                hipSafeCall(hipMemcpy(u->handle, u->origdata, u->size, hipMemcpyHostToDevice));
                u->markHostCopyObsolete(false);
                u->markDeviceCopyObsolete(false);
            }
        }
        return true;
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        CV_Assert(u->urefcount == 0 && u->refcount == 0);
        if (u->handle) {
            hipSafeCall(hipFree(u->handle));
            u->handle = nullptr;
        }
        if (u->data && !(u->flags & UMatData::USER_ALLOCATED)) {
            fastFree(u->data);
            u->data = nullptr;
        }
        delete u;
    }

    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        if (!u) return;
        UMatDataAutoLock lock(u);
        if (u->hostCopyObsolete() && u->handle) {
            if (!u->data)
                u->data = u->origdata = (uchar*)fastMalloc(u->size);
            hipSafeCall(hipMemcpy(u->data, u->handle, u->size, hipMemcpyDeviceToHost));
            u->markHostCopyObsolete(false);
        }
        if (accessFlags & ACCESS_WRITE)
            u->markDeviceCopyObsolete(true);
    }

    void unmap(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        UMatDataAutoLock lock(u);
        if (u->deviceCopyObsolete() && u->handle && u->data) {
            hipSafeCall(hipMemcpy(u->handle, u->data, u->size, hipMemcpyHostToDevice));
            u->markDeviceCopyObsolete(false);
            u->markHostCopyObsolete(true);
        }
        if (u->data && !(u->flags & UMatData::USER_ALLOCATED)) {
            fastFree(u->data);
            u->data     = nullptr;
            u->origdata = nullptr;
            u->markHostCopyObsolete(true);  // host buffer gone; device is authoritative
        }
    }

    void download(UMatData* u, void* dst, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const CV_OVERRIDE
    {
        if (!u || !u->handle) return;
        if (dims <= 2) {
            const uchar* src = (const uchar*)u->handle;
            if (dims == 2)
                src += srcofs[0] * srcstep[0] + srcofs[1];
            hipSafeCall(hipMemcpy2D(dst, dststep[0],
                                    src, srcstep[0],
                                    sz[dims - 1], sz[0],
                                    hipMemcpyDeviceToHost));
        } else {
            hipSafeCall(hipMemcpy(dst, u->handle, u->size, hipMemcpyDeviceToHost));
        }
    }

    void upload(UMatData* u, const void* src, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE
    {
        if (!u || !u->handle) return;
        if (dims <= 2) {
            uchar* dst = (uchar*)u->handle;
            if (dims == 2)
                dst += dstofs[0] * dststep[0] + dstofs[1];
            hipSafeCall(hipMemcpy2D(dst, dststep[0],
                                    src, srcstep[0],
                                    sz[dims - 1], sz[0],
                                    hipMemcpyHostToDevice));
        } else {
            hipSafeCall(hipMemcpy(u->handle, src, u->size, hipMemcpyHostToDevice));
        }
        u->markHostCopyObsolete(false);
        u->markDeviceCopyObsolete(false);
    }

    void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool sync) const CV_OVERRIDE
    {
        if (!srcdata || !dstdata) return;

        const bool srcOnDevice = (srcdata->handle != nullptr);
        const bool dstOnDevice = (dstdata->handle != nullptr);

        hipMemcpyKind kind;
        const void* rawSrc = nullptr;
        void*       rawDst = nullptr;

        if (srcOnDevice && dstOnDevice) {
            kind = hipMemcpyDeviceToDevice;
            rawSrc = srcdata->handle;
            rawDst = dstdata->handle;
        } else if (srcOnDevice) {
            kind = hipMemcpyDeviceToHost;
            rawSrc = srcdata->handle;
            rawDst = dstdata->data;
        } else if (dstOnDevice) {
            kind = hipMemcpyHostToDevice;
            rawSrc = srcdata->data;
            rawDst = dstdata->handle;
        } else {
            return; // both CPU — generic Mat path handles this
        }

        if (!rawSrc || !rawDst) return;

        if (dims <= 2) {
            const uchar* src = (const uchar*)rawSrc;
            uchar*       dst = (uchar*)rawDst;
            if (dims == 2) {
                src += srcofs[0] * srcstep[0] + srcofs[1];
                dst += dstofs[0] * dststep[0] + dstofs[1];
            }
            if (sync || kind != hipMemcpyDeviceToDevice)
                hipSafeCall(hipMemcpy2D(dst, dststep[0], src, srcstep[0],
                                        sz[dims - 1], sz[0], kind));
            else
                hipSafeCall(hipMemcpy2DAsync(dst, dststep[0], src, srcstep[0],
                                             sz[dims - 1], sz[0], kind, 0));
        } else {
            hipSafeCall(hipMemcpy(rawDst, rawSrc, srcdata->size, kind));
        }

        dstdata->markHostCopyObsolete(dstOnDevice);
        dstdata->markDeviceCopyObsolete(!dstOnDevice);
    }
};

HipAllocator hipAllocatorInstance;

} // anonymous namespace

namespace cv { namespace hip {

bool isHipUMat(InputArray a)
{
    if (!a.isUMat()) return false;
    UMat u = a.getUMat();
    return u.u != nullptr && u.u->currAllocator == getHipAllocator();
}

}} // cv::hip


namespace cv { namespace hip {

static bool g_useHip = true;

CV_EXPORTS_W bool useHip()
{
    if (!g_useHip) return false;
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
        g_useHip = false;
        return false;
    }
    // Disable OpenCL so its kernels never receive HIP-allocated buffers.
    // UMat paths check currAllocator for HIP, but Mat::copyTo / Mat::setTo
    // fire CV_OCL_RUN before that check and would pass a HIP buffer to OpenCL.
    cv::ocl::setUseOpenCL(false);
    return true;
}

CV_EXPORTS MatAllocator* getHipAllocator()
{
    return &hipAllocatorInstance;
}

}} // cv::hip
#endif

#ifndef HAVE_HIP
namespace cv { namespace hip {
bool isHipUMat(InputArray) { return false; }
}} // cv::hip
#endif

// ======================== Stream::Impl ========================

#ifdef HAVE_HIP
struct cv::hip::Stream::Impl
{
    hipStream_t stream;
    bool ownStream;

    Impl() : stream(0), ownStream(false) {}

    explicit Impl(unsigned int hipFlags)
        : stream(0), ownStream(true)
    {
        hipSafeCall(hipStreamCreateWithFlags(&stream, hipFlags));
    }

    explicit Impl(hipStream_t s) : stream(s), ownStream(false) {}

    ~Impl()
    {
        if (ownStream && stream) hipSafeCall(hipStreamDestroy(stream));
    }
};
#endif

// ======================== Stream ========================

cv::hip::Stream::Stream()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    impl = makePtr<Impl>();
#endif
}

cv::hip::Stream::Stream(const size_t hipFlags_)
{
#ifndef HAVE_HIP
    CV_UNUSED(hipFlags_); throw_no_hip();
#else
    impl = makePtr<Impl>((unsigned int)hipFlags_);
#endif
}

cv::hip::Stream::Stream(const Ptr<Impl>& impl_) : impl(impl_) {}

bool cv::hip::Stream::queryIfComplete() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipError_t err = hipStreamQuery(impl->stream);
    if (err == hipSuccess) return true;
    if (err == hipErrorNotReady) return false;
    hipSafeCall(err);
    return false;
#endif
}

void cv::hip::Stream::waitForCompletion()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipSafeCall(hipStreamSynchronize(impl->stream));
#endif
}

void cv::hip::Stream::enqueueHostCallback(StreamCallback callback, void* userData)
{
#ifndef HAVE_HIP
    CV_UNUSED(callback); CV_UNUSED(userData); throw_no_hip();
#else
    hipSafeCall(hipStreamAddCallback(impl->stream,
        reinterpret_cast<hipStreamCallback_t>(callback), userData, 0));
#endif
}

Stream& cv::hip::Stream::Null()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    static Stream s(makePtr<Stream::Impl>());
    return s;
#endif
}

cv::hip::Stream::operator bool_type() const
{
#ifndef HAVE_HIP
    return nullptr;
#else
    return (impl && impl->stream != 0) ?
        &Stream::this_type_does_not_support_comparisions : nullptr;
#endif
}

void* cv::hip::Stream::hipPtr() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    return impl->stream;
#endif
}

// ======================== StreamAccessor ========================

hipStream_t cv::hip::StreamAccessor::getStream(const Stream& stream)
{
#ifndef HAVE_HIP
    CV_UNUSED(stream); throw_no_hip();
#else
    return stream.impl->stream;
#endif
}

Stream cv::hip::StreamAccessor::wrapStream(hipStream_t stream_)
{
#ifndef HAVE_HIP
    CV_UNUSED(stream_); throw_no_hip();
#else
    return Stream(makePtr<Stream::Impl>(stream_));
#endif
}

Stream cv::hip::wrapStream(size_t hipStreamMemoryAddress)
{
#ifndef HAVE_HIP
    CV_UNUSED(hipStreamMemoryAddress); throw_no_hip();
#else
    return StreamAccessor::wrapStream(reinterpret_cast<hipStream_t>(hipStreamMemoryAddress));
#endif
}

// ======================== Event::Impl ========================

#ifdef HAVE_HIP
struct cv::hip::Event::Impl
{
    hipEvent_t event;

    explicit Impl(unsigned int flags_)
    {
        hipSafeCall(hipEventCreateWithFlags(&event, flags_));
    }

    ~Impl()
    {
        if (event) hipSafeCall(hipEventDestroy(event));
    }
};
#endif

// ======================== Event ========================

cv::hip::Event::Event(const Event::CreateFlags flags_)
{
#ifndef HAVE_HIP
    CV_UNUSED(flags_); throw_no_hip();
#else
    unsigned int hf = 0;
    if (flags_ & BLOCKING_SYNC)  hf |= hipEventBlockingSync;
    if (flags_ & DISABLE_TIMING) hf |= hipEventDisableTiming;
    if (flags_ & INTERPROCESS)   hf |= hipEventInterprocess;
    impl_ = makePtr<Impl>(hf);
#endif
}

void cv::hip::Event::record(Stream& stream_)
{
#ifndef HAVE_HIP
    CV_UNUSED(stream_); throw_no_hip();
#else
    hipSafeCall(hipEventRecord(impl_->event, StreamAccessor::getStream(stream_)));
#endif
}

bool cv::hip::Event::queryIfComplete() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipError_t err = hipEventQuery(impl_->event);
    if (err == hipSuccess) return true;
    if (err == hipErrorNotReady) return false;
    hipSafeCall(err);
    return false;
#endif
}

void cv::hip::Event::waitForCompletion()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipSafeCall(hipEventSynchronize(impl_->event));
#endif
}

float cv::hip::Event::elapsedTime(const Event& start, const Event& end)
{
#ifndef HAVE_HIP
    CV_UNUSED(start); CV_UNUSED(end); throw_no_hip();
#else
    float ms = 0.0f;
    hipSafeCall(hipEventElapsedTime(&ms, start.impl_->event, end.impl_->event));
    return ms;
#endif
}

// ======================== EventAccessor ========================

hipEvent_t cv::hip::EventAccessor::getEvent(const Event& event)
{
#ifndef HAVE_HIP
    CV_UNUSED(event); throw_no_hip();
#else
    return event.impl_->event;
#endif
}

Event cv::hip::EventAccessor::wrapEvent(hipEvent_t event_)
{
#ifndef HAVE_HIP
    CV_UNUSED(event_); throw_no_hip();
#else
    CV_UNUSED(event_);
    CV_Error(cv::Error::StsNotImplemented, "EventAccessor::wrapEvent not supported");
#endif
}

// ======================== Device management ========================

int cv::hip::getHipEnabledDeviceCount()
{
#ifndef HAVE_HIP
    return 0;
#else
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err == hipErrorNoDevice)           return 0;
    if (err == hipErrorInsufficientDriver) return -1;
    hipSafeCall(err);
    return count;
#endif
}

void cv::hip::setDevice(int device)
{
#ifndef HAVE_HIP
    CV_UNUSED(device); throw_no_hip();
#else
    hipSafeCall(hipSetDevice(device));
#endif
}

int cv::hip::getDevice()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    int device = 0;
    hipSafeCall(hipGetDevice(&device));
    return device;
#endif
}

void cv::hip::resetDevice()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipSafeCall(hipDeviceReset());
#endif
}

bool cv::hip::deviceSupports(FeatureSet feature_set)
{
#ifndef HAVE_HIP
    CV_UNUSED(feature_set); return false;
#else
    if (getHipEnabledDeviceCount() <= 0) return false;
    hipDeviceProp_t prop;
    hipSafeCall(hipGetDeviceProperties(&prop, getDevice()));
    switch (feature_set) {
        case GLOBAL_ATOMICS:         return prop.arch.hasGlobalInt32Atomics != 0;
        case SHARED_ATOMICS:         return prop.arch.hasSharedInt32Atomics != 0;
        case NATIVE_DOUBLE:          return prop.arch.hasDoubles != 0;
        case WARP_SHUFFLE_FUNCTIONS: return prop.arch.hasWarpShuffle != 0;
        case DYNAMIC_PARALLELISM:    return prop.arch.hasDynamicParallelism != 0;
    }
    return false;
#endif
}

// ======================== TargetArchs ========================

bool cv::hip::TargetArchs::builtWith(FeatureSet feature_set) { return deviceSupports(feature_set); }

bool cv::hip::TargetArchs::has(int major, int minor)
{
#ifndef HAVE_HIP
    CV_UNUSED(major); CV_UNUSED(minor); return false;
#else
    int count = getHipEnabledDeviceCount();
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p; hipSafeCall(hipGetDeviceProperties(&p, i));
        if (p.major == major && p.minor == minor) return true;
    }
    return false;
#endif
}

bool cv::hip::TargetArchs::hasBin(int major, int minor) { return has(major, minor); }

bool cv::hip::TargetArchs::hasEqualOrGreater(int major, int minor)
{
#ifndef HAVE_HIP
    CV_UNUSED(major); CV_UNUSED(minor); return false;
#else
    int count = getHipEnabledDeviceCount(), ref = major * 10 + minor;
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p; hipSafeCall(hipGetDeviceProperties(&p, i));
        if (p.major * 10 + p.minor >= ref) return true;
    }
    return false;
#endif
}

bool cv::hip::TargetArchs::hasEqualOrGreaterBin(int major, int minor)
{
    return hasEqualOrGreater(major, minor);
}

// ======================== DeviceInfo ========================

#ifdef HAVE_HIP
static hipDeviceProp_t getDeviceProp(int id)
{
    hipDeviceProp_t p;
    hipSafeCall(hipGetDeviceProperties(&p, id));
    return p;
}
#endif

cv::hip::DeviceInfo::DeviceInfo()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipSafeCall(hipGetDevice(&device_id_));
#endif
}

cv::hip::DeviceInfo::DeviceInfo(int device_id) : device_id_(device_id)
{
#ifndef HAVE_HIP
    CV_UNUSED(device_id); throw_no_hip();
#else
    CV_Assert(device_id_ >= 0 && device_id_ < getHipEnabledDeviceCount());
#endif
}

int cv::hip::DeviceInfo::deviceID() const { return device_id_; }

const char* cv::hip::DeviceInfo::name() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    static thread_local char buf[256];
    std::strncpy(buf, getDeviceProp(device_id_).name, 255);
    return buf;
#endif
}

#define DEVINFO_PROP(rettype, method, field) \
rettype cv::hip::DeviceInfo::method() const { \
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip(); \
    return (rettype)getDeviceProp(device_id_).field; \
}

DEVINFO_PROP(size_t, totalGlobalMem,           totalGlobalMem)
DEVINFO_PROP(size_t, sharedMemPerBlock,        sharedMemPerBlock)
DEVINFO_PROP(int,    regsPerBlock,             regsPerBlock)
DEVINFO_PROP(int,    warpSize,                 warpSize)
DEVINFO_PROP(size_t, memPitch,                 memPitch)
DEVINFO_PROP(int,    maxThreadsPerBlock,       maxThreadsPerBlock)
DEVINFO_PROP(int,    clockRate,                clockRate)
DEVINFO_PROP(size_t, totalConstMem,            totalConstMem)
DEVINFO_PROP(int,    majorVersion,             major)
DEVINFO_PROP(int,    minorVersion,             minor)
DEVINFO_PROP(size_t, textureAlignment,         textureAlignment)
DEVINFO_PROP(size_t, texturePitchAlignment,    texturePitchAlignment)
DEVINFO_PROP(int,    multiProcessorCount,      multiProcessorCount)
DEVINFO_PROP(int,    memoryClockRate,          memoryClockRate)
DEVINFO_PROP(int,    memoryBusWidth,           memoryBusWidth)
DEVINFO_PROP(int,    l2CacheSize,              l2CacheSize)
DEVINFO_PROP(int,    maxThreadsPerMultiProcessor, maxThreadsPerMultiProcessor)
DEVINFO_PROP(int,    maxTexture1D,             maxTexture1D)
DEVINFO_PROP(int,    maxTexture1DLinear,        maxTexture1DLinear)
DEVINFO_PROP(int,    pciBusID,                 pciBusID)
DEVINFO_PROP(int,    pciDeviceID,              pciDeviceID)
DEVINFO_PROP(int,    pciDomainID,              pciDomainID)
DEVINFO_PROP(int,    asicRevision,             asicRevision)

bool cv::hip::DeviceInfo::kernelExecTimeoutEnabled() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).kernelExecTimeoutEnabled != 0;
}
bool cv::hip::DeviceInfo::integrated() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).integrated != 0;
}
bool cv::hip::DeviceInfo::canMapHostMemory() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).canMapHostMemory != 0;
}
bool cv::hip::DeviceInfo::concurrentKernels() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).concurrentKernels != 0;
}
bool cv::hip::DeviceInfo::ECCEnabled() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).ECCEnabled != 0;
}
bool cv::hip::DeviceInfo::tccDriver() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).tccDriver != 0;
}
bool cv::hip::DeviceInfo::cooperativeLaunch() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).cooperativeLaunch != 0;
}
bool cv::hip::DeviceInfo::isLargeBar() const
{
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).isLargeBar != 0;
}

DeviceInfo::ComputeMode cv::hip::DeviceInfo::computeMode() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    switch (getDeviceProp(device_id_).computeMode) {
        case 0: return ComputeModeDefault;
        case 1: return ComputeModeExclusive;
        case 2: return ComputeModeProhibited;
        case 3: return ComputeModeExclusiveProcess;
        default: return ComputeModeDefault;
    }
#endif
}

Vec3i cv::hip::DeviceInfo::maxThreadsDim() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec3i(p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
#endif
}

Vec3i cv::hip::DeviceInfo::maxGridSize() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec3i(p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
#endif
}

Vec2i cv::hip::DeviceInfo::maxTexture2D() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec2i(p.maxTexture2D[0], p.maxTexture2D[1]);
#endif
}

Vec3i cv::hip::DeviceInfo::maxTexture3D() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec3i(p.maxTexture3D[0], p.maxTexture3D[1], p.maxTexture3D[2]);
#endif
}

const char* cv::hip::DeviceInfo::gcnArchName() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    static thread_local char buf[256];
    std::strncpy(buf, getDeviceProp(device_id_).gcnArchName, 255);
    return buf;
#endif
}

void cv::hip::DeviceInfo::queryMemory(size_t& totalMemory, size_t& freeMemory) const
{
#ifndef HAVE_HIP
    CV_UNUSED(totalMemory); CV_UNUSED(freeMemory); throw_no_hip();
#else
    int prev = getDevice();
    if (prev != device_id_) setDevice(device_id_);
    hipSafeCall(hipMemGetInfo(&freeMemory, &totalMemory));
    if (prev != device_id_) setDevice(prev);
#endif
}

size_t cv::hip::DeviceInfo::freeMemory()  const { size_t t = 0, f = 0; queryMemory(t, f); return f; }
size_t cv::hip::DeviceInfo::totalMemory() const { size_t t = 0, f = 0; queryMemory(t, f); return t; }

bool cv::hip::DeviceInfo::supports(FeatureSet feature_set) const
{
#ifndef HAVE_HIP
    CV_UNUSED(feature_set); return false;
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    switch (feature_set) {
        case GLOBAL_ATOMICS:         return p.arch.hasGlobalInt32Atomics != 0;
        case SHARED_ATOMICS:         return p.arch.hasSharedInt32Atomics != 0;
        case NATIVE_DOUBLE:          return p.arch.hasDoubles != 0;
        case WARP_SHUFFLE_FUNCTIONS: return p.arch.hasWarpShuffle != 0;
        case DYNAMIC_PARALLELISM:    return p.arch.hasDynamicParallelism != 0;
    }
    return false;
#endif
}

bool cv::hip::DeviceInfo::isCompatible() const
{
#ifndef HAVE_HIP
    return false;
#else
    return getHipEnabledDeviceCount() > 0 && device_id_ < getHipEnabledDeviceCount();
#endif
}

// ======================== Print functions ========================

void cv::hip::printHipDeviceInfo(int device)
{
#ifndef HAVE_HIP
    CV_UNUSED(device); throw_no_hip();
#else
    hipDeviceProp_t p;
    hipSafeCall(hipGetDeviceProperties(&p, device));
    std::printf("Device %d: \"%s\"\n",                     device, p.name);
    std::printf("  HIP Compute Capability:              %d.%d\n", p.major, p.minor);
    std::printf("  GCN Architecture:                    %s\n",    p.gcnArchName);
    std::printf("  Total global memory:                 %.0f MB\n", (double)p.totalGlobalMem / (1 << 20));
    std::printf("  Shared memory per block:             %zu bytes\n", p.sharedMemPerBlock);
    std::printf("  Registers per block:                 %d\n",    p.regsPerBlock);
    std::printf("  Warp size:                           %d\n",    p.warpSize);
    std::printf("  Max threads per block:               %d\n",    p.maxThreadsPerBlock);
    std::printf("  Max block dimensions:                [%d, %d, %d]\n",
                p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    std::printf("  Max grid dimensions:                 [%d, %d, %d]\n",
                p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    std::printf("  Clock rate:                          %.2f GHz\n", p.clockRate * 1e-6);
    std::printf("  Memory clock rate:                   %.2f GHz\n", p.memoryClockRate * 1e-6);
    std::printf("  Memory bus width:                    %d-bit\n", p.memoryBusWidth);
    std::printf("  L2 cache size:                       %d bytes\n", p.l2CacheSize);
    std::printf("  Multiprocessors:                     %d\n",    p.multiProcessorCount);
    std::printf("  Max threads per multiprocessor:      %d\n",    p.maxThreadsPerMultiProcessor);
    std::printf("  Concurrent kernels:                  %s\n",    p.concurrentKernels ? "Yes" : "No");
    std::printf("  ECC enabled:                         %s\n",    p.ECCEnabled ? "Yes" : "No");
    std::printf("  Cooperative launch:                  %s\n",    p.cooperativeLaunch ? "Yes" : "No");
    std::printf("  Large bar:                           %s\n",    p.isLargeBar ? "Yes" : "No");
    std::printf("  PCI Bus/Device/Domain:               %d/%d/%d\n",
                p.pciBusID, p.pciDeviceID, p.pciDomainID);
#endif
}

void cv::hip::printShortHipDeviceInfo(int device)
{
#ifndef HAVE_HIP
    CV_UNUSED(device); throw_no_hip();
#else
    hipDeviceProp_t p;
    hipSafeCall(hipGetDeviceProperties(&p, device));
    std::printf("Device %d: \"%s\"  %.0f MB  compute %d.%d  %s\n",
                device, p.name,
                (double)p.totalGlobalMem / (1 << 20),
                p.major, p.minor, p.gcnArchName);
#endif
}
