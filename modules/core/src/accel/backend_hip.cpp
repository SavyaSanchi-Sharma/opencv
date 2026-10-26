// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the HIP backend (AMD GPUs), built as a plugin with
// modules/core/misc/plugins/accel_hip. OpenCV's own build compiles this file empty.
//
// Discrete GPUs: device memory from the stream-ordered pool (hipMallocAsync), which keeps freed
// memory for reuse; the CPU works on a pinned copy, so uploads from it run asynchronously.
// APUs (integrated GPUs): pinned host memory mapped into the GPU, which the CPU and the GPU
// both address, so nothing is copied.

#ifdef HAVE_ACCEL_HIP

#include "opencv2/core/accel_backend.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "hip/accel_kernels_hip.hpp"

#include <hip/hip_runtime_api.h>

#include <memory>
#include <mutex>
#include <vector>

namespace cv { namespace accel { namespace hip_impl {

namespace {

void check(hipError_t e, const char* what)
{
    if (e == hipSuccess)
        return;
    (void)hipGetLastError();    // a non-sticky error must not reach the thread's next call
    CV_Error_(Error::GpuApiCallError, ("accel(hip): %s failed: %s", what, hipGetErrorString(e)));
}

//! makes @p device current for the calling thread while it lives
class DeviceGuard
{
public:
    explicit DeviceGuard(int device) : prev_(-1)
    {
        int current = -1;
        check(hipGetDevice(&current), "hipGetDevice");
        if (current != device)
        {
            check(hipSetDevice(device), "hipSetDevice");
            prev_ = current;
        }
    }
    ~DeviceGuard()
    {
        if (prev_ >= 0)
            (void)hipSetDevice(prev_);
    }
private:
    int prev_;
};

class HipEvent CV_FINAL : public Event
{
public:
    explicit HipEvent(hipEvent_t e) : e_(e) {}
    ~HipEvent() { (void)hipEventDestroy(e_); }

    bool query() CV_OVERRIDE
    {
        const hipError_t r = hipEventQuery(e_);
        if (r == hipErrorNotReady)
        {
            (void)hipGetLastError();
            return false;
        }
        check(r, "hipEventQuery");
        return true;
    }

    void wait() CV_OVERRIDE
    {
        check(hipEventSynchronize(e_), "hipEventSynchronize");
    }

    hipEvent_t e_;
};

class HipStream CV_FINAL : public Stream
{
public:
    explicit HipStream(int device) : device_(device), s_(0)
    {
        DeviceGuard g(device);
        check(hipStreamCreateWithFlags(&s_, hipStreamNonBlocking), "hipStreamCreateWithFlags");
    }
    ~HipStream() { (void)hipStreamDestroy(s_); }   // queued work still completes

    Ptr<Event> record() CV_OVERRIDE
    {
        DeviceGuard g(device_);
        hipEvent_t e = 0;
        check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "hipEventCreateWithFlags");
        const hipError_t r = hipEventRecord(e, s_);
        if (r != hipSuccess)
        {
            (void)hipEventDestroy(e);
            check(r, "hipEventRecord");
        }
        return makePtr<HipEvent>(e);
    }

    bool waitEvent(const Ptr<Event>& event) CV_OVERRIDE
    {
        HipEvent* he = dynamic_cast<HipEvent*>(event.get());
        if (!he)
            return false;
        DeviceGuard g(device_);
        check(hipStreamWaitEvent(s_, he->e_, 0), "hipStreamWaitEvent");
        return true;
    }

    void synchronize() CV_OVERRIDE
    {
        check(hipStreamSynchronize(s_), "hipStreamSynchronize");
    }

    const int device_;
    hipStream_t s_;
};

HipStream& hipStreamOf(Stream& s)
{
    HipStream* hs = dynamic_cast<HipStream*>(&s);
    CV_Assert(hs && "accel(hip): the stream belongs to another backend");
    return *hs;
}

class HipCoreOps CV_FINAL : public CoreOps
{
public:
    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !kernels::supportedElementSize(esz))
            return false;
        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::flip((const unsigned char*)devicePtr(src), src.step[0],
                            (unsigned char*)devicePtr(dst), dst.step[0],
                            src.rows, src.cols, esz, flipCode, s.s_), "flip");
        return true;
    }

    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !kernels::supportedElementSize(esz))
            return false;
        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::transpose((const unsigned char*)devicePtr(src), src.step[0],
                                 (unsigned char*)devicePtr(dst), dst.step[0],
                                 src.rows, src.cols, esz, s.s_), "transpose");
        return true;
    }
};

class HipBackend CV_FINAL : public Backend
{
public:
    //! a device attribute, 0 when the runtime will not tell us
    static int attr(hipDeviceAttribute_t a, int device)
    {
        int v = 0;
        if (hipDeviceGetAttribute(&v, a, device) != hipSuccess)
        {
            (void)hipGetLastError();
            return 0;
        }
        return v;
    }

    HipBackend()
    {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess)
        {
            (void)hipGetLastError();    // no driver or no device
            n = 0;
        }
        for (int i = 0; i < n; i++)
        {
            std::unique_ptr<Dev> d(new Dev());
            check(hipGetDeviceProperties(&d->prop, i), "hipGetDeviceProperties");
            d->shared = d->prop.integrated && d->prop.canMapHostMemory;
            /* What this device can provide. Device memory always; host memory the GPU can
               address only where the two share it, which on HIP means an integrated part;
               managed memory where the runtime reports it, discrete parts included. */
            d->kinds = MEMORY_DEVICE;
            if (d->shared)
                d->kinds |= MEMORY_SHARED_COHERENT | MEMORY_SHARED_NONCOHERENT;
            /* Managed memory, and what XNACK gives when it is on. With XNACK the GPU retries
               a faulting access instead of dying, so it can reach pageable host memory and
               managed memory migrates on demand rather than being pinned up front. The arch
               name carries the setting as a target id suffix, gfx1100:xnack+ or :xnack-.
               Nothing here depends on it yet; it decides how good MEMORY_MANAGED actually is,
               which is a question for the promotion rule and for measurements. */
            d->managed = attr(hipDeviceAttributeManagedMemory, i);
            d->pageableAccess = attr(hipDeviceAttributePageableMemoryAccess, i);
            d->concurrentManaged = attr(hipDeviceAttributeConcurrentManagedAccess, i);
            if (d->managed)
                d->kinds |= MEMORY_MANAGED;
            CV_LOG_INFO(NULL, "accel hip:" << i << " " << d->prop.gcnArchName
                        << " integrated=" << (int)d->prop.integrated
                        << " kinds=" << d->kinds
                        << " managed=" << d->managed
                        << " pageable(xnack)=" << d->pageableAccess
                        << " concurrentManaged=" << d->concurrentManaged);
            int pools = 0;
            if (hipDeviceGetAttribute(&pools, hipDeviceAttributeMemoryPoolsSupported, i) != hipSuccess)
            {
                (void)hipGetLastError();
                pools = 0;
            }
            d->pools = pools != 0 && !d->shared;
            devs_.push_back(std::move(d));
        }
    }

    String name() const CV_OVERRIDE { return "hip"; }
    int deviceCount() const CV_OVERRIDE { return (int)devs_.size(); }
    String deviceDescription(int device) const CV_OVERRIDE { return devs_[device]->prop.name; }
    int memoryKinds(int device) const CV_OVERRIDE { return devs_[device]->kinds; }

    /* A discrete card has only its own memory. An integrated one has a choice, and the
       answer depends on what the buffer is for: measured on gfx1036, a coalesced kernel
       reads both kinds at the same speed, an awkward one (transpose) runs 2.5-3.6x faster
       in device memory, and pinning host memory costs about 0.2 ms per MB. So device memory
       is the better guess for a buffer the CPU never touches, and the promotion in
       DeviceAllocator::map() gives the zero-copy back to buffers it does touch.

       Off by default: promotion has to be enabled for this to pay, and the threshold it
       needs is a per-machine number. OPENCV_ACCEL_APU_DEVICE_DEFAULT=1 with
       OPENCV_ACCEL_ZERO_COPY_MAX_BYTES set turns the pair on. */
    MemoryKind defaultMemoryKind(int device) const CV_OVERRIDE
    {
        if (!devs_[device]->shared)
            return MEMORY_DEVICE;
        static const bool deviceDefault =
            utils::getConfigurationParameterBool("OPENCV_ACCEL_APU_DEVICE_DEFAULT", false);
        return deviceDefault ? MEMORY_DEVICE : MEMORY_SHARED_COHERENT;
    }

    void* allocate(int device, size_t size, MemoryKind kind) CV_OVERRIDE
    {
        Dev& d = *devs_[device];
        CV_Assert((d.kinds & (int)kind) != 0 && "accel(hip): a kind this device does not provide");
        DeviceGuard g(device);
        void* p = 0;
        hipError_t r;
        switch (kind)
        {
        case MEMORY_SHARED_COHERENT:
            // one address for the CPU and the GPU
            r = hipHostMalloc(&p, size, hipHostMallocMapped | hipHostMallocPortable);
            break;
        case MEMORY_SHARED_NONCOHERENT:
            /* As above but cached on the CPU, so it is faster to read and write there.
               beginCpuAccess/endCpuAccess are left as the base no-ops: HIP makes
               coarse-grained memory consistent at command boundaries, and core waits for the
               last device writer before it calls beginCpuAccess, so there should be nothing
               to flush. TODO(apu-validation): only an APU can prove that; until one runs the
               tests, prefer MEMORY_SHARED_COHERENT. */
            r = hipHostMalloc(&p, size,
                              hipHostMallocMapped | hipHostMallocPortable | hipHostMallocNonCoherent);
            break;
        case MEMORY_MANAGED:
            r = hipMallocManaged(&p, size, hipMemAttachGlobal);
            break;
        default:
            if (d.pools)
            {
                hipStream_t s = allocStream(device);
                r = hipMallocAsync(&p, size, s);
                if (r == hipSuccess)
                    r = hipStreamSynchronize(s);    // from now on usable on every stream
            }
            else
                r = hipMalloc(&p, size);
            break;
        }
        if (r == hipErrorOutOfMemory)
        {
            (void)hipGetLastError();
            return 0;
        }
        check(r, "allocate");
        if (isSharedKind(kind))
        {
            // the backend hands the CPU address straight back, so the two must agree
            void* dp = 0;
            if (hipHostGetDevicePointer(&dp, p, 0) != hipSuccess)
                (void)hipGetLastError();
            else if (dp != p)
            {
                static bool warned = false;
                if (!warned)
                {
                    warned = true;
                    CV_LOG_WARNING(NULL, "accel(hip): the device address of shared memory differs from "
                                   "the host address; this backend assumes unified addressing");
                }
            }
        }
        return p;
    }

    void deallocate(int device, void* ptr, MemoryKind kind) CV_OVERRIDE
    {
        // called while UMats are released: report failures, never throw
        try
        {
            Dev& d = *devs_[device];
            DeviceGuard g(device);
            // freed the way it was allocated; hipFree takes managed memory
            const hipError_t r = isSharedKind(kind) && kind != MEMORY_MANAGED ? hipHostFree(ptr)
                               : kind == MEMORY_MANAGED ? hipFree(ptr)
                               : d.pools ? hipFreeAsync(ptr, allocStream(device))
                               : hipFree(ptr);
            if (r != hipSuccess)
            {
                (void)hipGetLastError();
                CV_LOG_WARNING(NULL, "accel(hip): freeing device memory failed: " << hipGetErrorString(r));
            }
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_WARNING(NULL, "accel(hip): freeing device memory failed: " << e.what());
        }
    }

    void* allocateHost(int device, size_t size) CV_OVERRIDE
    {
        DeviceGuard g(device);
        void* p = 0;
        if (hipHostMalloc(&p, size, hipHostMallocPortable) != hipSuccess)
        {
            (void)hipGetLastError();
            return 0;
        }
        return p;
    }

    void deallocateHost(int, void* ptr) CV_OVERRIDE
    {
        if (hipHostFree(ptr) != hipSuccess)
            (void)hipGetLastError();
    }

    void copy(int device, CopyKind kind, void* dst, size_t dstStep, const void* src, size_t srcStep,
              size_t widthBytes, size_t rows, Stream& stream, bool hostStable) CV_OVERRIDE
    {
        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(device);
        check(hipMemcpy2DAsync(dst, dstStep, src, srcStep, widthBytes, rows, hipMemcpyDefault, s.s_),
              "hipMemcpy2DAsync");
        if (kind == COPY_HOST_TO_DEVICE && !hostStable)
        {
            // the caller's memory may change once this returns, and pinned memory is read asynchronously
            Ptr<Event> e = s.record();
            e->wait();
        }
    }

    Ptr<Stream> createStream(int device) CV_OVERRIDE
    {
        return makePtr<HipStream>(device);
    }

    Ops* ops(const String& module) CV_OVERRIDE
    {
        return module == CoreOps::module() ? &core_ : 0;
    }

private:
    struct Dev
    {
        Dev() : shared(false), kinds(MEMORY_DEVICE), managed(0), pageableAccess(0),
                concurrentManaged(0), pools(false), allocStream_(0) {}
        hipDeviceProp_t prop;
        bool shared;
        int kinds;              // bitmask of MemoryKind
        int managed;            // hipMallocManaged works
        int pageableAccess;     // the GPU reaches pageable host memory, so XNACK is on
        int concurrentManaged;  // CPU and GPU may touch managed memory at the same time
        bool pools;
        std::once_flag once;
        hipStream_t allocStream_;
    };

    //! where pool memory is allocated and freed; created with the device's context on first use
    hipStream_t allocStream(int device)
    {
        Dev& d = *devs_[device];
        std::call_once(d.once, [&]() {
            DeviceGuard g(device);
            hipMemPool_t pool;
            if (hipDeviceGetDefaultMemPool(&pool, device) == hipSuccess)
            {
                // keep freed memory in the pool instead of returning it to the driver at each synchronization
                uint64_t threshold = UINT64_MAX;
                if (hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold) != hipSuccess)
                    (void)hipGetLastError();
            }
            else
                (void)hipGetLastError();
            check(hipStreamCreateWithFlags(&d.allocStream_, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        });
        return d.allocStream_;
    }

    std::vector<std::unique_ptr<Dev> > devs_;
    HipCoreOps core_;
};

} // namespace

#ifndef BUILD_PLUGIN
//! built into core: the registry creates it (registry.cpp), there is no plugin to load
Ptr<Backend> createHipBackend()
{
    return makePtr<HipBackend>();
}
#endif

}}} // namespace cv::accel::hip_impl

#ifdef BUILD_PLUGIN
CV_ACCEL_DEFINE_PLUGIN(new cv::accel::hip_impl::HipBackend())
#endif

#endif // HAVE_ACCEL_HIP
