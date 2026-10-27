// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the Metal backend (Apple GPUs). Built into core on Apple platforms where
// HAVE_METAL was detected, and standalone as a plugin with modules/core/misc/plugins/accel_metal.
//
// Apple silicon: the CPU and the GPU share one pool of DRAM, so a Shared buffer is reached by
// both without a copy and that is the default. Private memory is still worth having for
// intermediates the CPU never looks at.
// Intel Macs with a discrete GPU: the GPU has its own memory and Managed buffers keep a copy on
// each side, synchronized explicitly. That is exactly the shape of the beginCpuAccess /
// endCpuAccess bracket, so MEMORY_SHARED_NONCOHERENT maps onto it.
//
// Unlike CUDA and HIP, a Metal buffer is an object and not an address: nothing here can add an
// offset to a handle. That is why the interface passes DeviceRef and not void*.

#ifdef HAVE_ACCEL_METAL

#include "opencv2/core/accel_backend.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "metal/accel_kernels_metal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <TargetConditionals.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <stdlib.h>
#include <unistd.h>

#if __has_feature(objc_arc)
#  error "accel(metal): build this file without ARC (-fno-objc-arc); it retains and releases by hand"
#endif

namespace cv { namespace accel { namespace metal_impl {

namespace {

//! macOS is the only platform with MTLStorageModeManaged; elsewhere there is nothing to manage
#if TARGET_OS_OSX
#  define CV_METAL_HAS_MANAGED 1
#else
#  define CV_METAL_HAS_MANAGED 0
#endif

std::string describe(NSError* e)
{
    if (!e)
        return "unknown error";
    const char* s = [[e localizedDescription] UTF8String];
    return s ? std::string(s) : std::string("unknown error");
}

/* The failure of work already queued. A command buffer reports its error long after the call
   that queued it returned, so the error is parked here and thrown by the next query(), wait()
   or synchronize() that looks, the way a sticky CUDA error would surface. Shared by a stream
   and every event it hands out, so an event outliving its stream still reports. */
class ErrorBox
{
public:
    void set(const std::string& what)
    {
        std::lock_guard<std::mutex> lock(m_);
        if (what_.empty())      // the first failure is the interesting one
            what_ = what;
    }

    void check() const
    {
        std::string what;
        {
            std::lock_guard<std::mutex> lock(m_);
            what = what_;
        }
        if (!what.empty())
            CV_Error_(Error::GpuApiCallError, ("accel(metal): device work failed: %s", what.c_str()));
    }

private:
    mutable std::mutex m_;
    std::string what_;
};

typedef std::shared_ptr<ErrorBox> ErrorBoxPtr;

/* An MTLSharedEvent counter value. The event is the stream's, shared by every point recorded
   on it; this object is one value of it, which the GPU reaches in order. */
class MetalEvent CV_FINAL : public Event
{
public:
    MetalEvent(id<MTLSharedEvent> e, uint64_t value, const ErrorBoxPtr& err)
        : e_([e retain]), value_(value), err_(err) {}
    ~MetalEvent() { [e_ release]; }

    bool query() CV_OVERRIDE
    {
        const bool done = e_.signaledValue >= value_;
        err_->check();
        return done;
    }

    void wait() CV_OVERRIDE
    {
        if (e_.signaledValue < value_)
        {
            // a long timeout rather than none: a wedged GPU should surface as an error, not a hang
            const uint64_t timeoutMs = 60ULL * 60ULL * 1000ULL;
            if (![e_ waitUntilSignaledValue:value_ timeoutMS:timeoutMs])
            {
                err_->check();      // a failure explains the wait better than the timeout does
                CV_Error(Error::GpuApiCallError, "accel(metal): timed out waiting for device work");
            }
        }
        err_->check();
    }

    id<MTLSharedEvent> e_;
    const uint64_t value_;
    ErrorBoxPtr err_;
};

/* A command queue and the command buffer being filled. Operations encode into that buffer and
   it is committed when a point in the stream is asked for, so several operations travel to the
   GPU together instead of one submission each. Command buffers of one queue begin in the order
   they were committed, which is the in-order queue the interface describes. */
class MetalStream CV_FINAL : public Stream
{
public:
    MetalStream(int device, id<MTLDevice> dev)
        : device_(device), err_(std::make_shared<ErrorBox>()), cb_(nil), last_(nil), counter_(0)
    {
        q_ = [dev newCommandQueue];
        if (!q_)
            CV_Error(Error::GpuApiCallError, "accel(metal): could not create a command queue");
        ev_ = [dev newSharedEvent];
        if (!ev_)
        {
            [q_ release];
            CV_Error(Error::GpuApiCallError, "accel(metal): could not create a shared event");
        }
    }

    ~MetalStream()
    {
        // queued work still completes; the queue outlives this object until the GPU is done
        [last_ release];
        [cb_ release];
        [ev_ release];
        [q_ release];
    }

    //! the command buffer to encode into, opening one if none is. The caller holds m_.
    id<MTLCommandBuffer> bufferLocked()
    {
        if (!cb_)
        {
            cb_ = [[q_ commandBuffer] retain];
            if (!cb_)
                CV_Error(Error::GpuApiCallError, "accel(metal): could not create a command buffer");
            ErrorBoxPtr err = err_;     // the block keeps the box alive past this stream
            [cb_ addCompletedHandler:^(id<MTLCommandBuffer> done) {
                if (done.error)
                    err->set(describe(done.error));
            }];
        }
        return cb_;
    }

    //! sends what has been encoded so far. The caller holds m_.
    void commitLocked()
    {
        if (!cb_)
            return;
        [cb_ commit];
        [last_ release];
        last_ = cb_;        // kept so synchronize() has something to wait on
        cb_ = nil;
    }

    Ptr<Event> record() CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(m_);
        if (cb_)
        {
            // the value this buffer signals on completion; later buffers signal larger ones
            [bufferLocked() encodeSignalEvent:ev_ value:++counter_];
            commitLocked();
        }
        /* With nothing open, counter_ is the value the last committed buffer signals, so an
           event on it completes when that buffer does. With nothing ever committed it is 0,
           which the event already reads, and the event is complete from the start. */
        return makePtr<MetalEvent>(ev_, counter_, err_);
    }

    bool waitEvent(const Ptr<Event>& event) CV_OVERRIDE
    {
        MetalEvent* me = dynamic_cast<MetalEvent*>(event.get());
        if (!me)
            return false;       // another backend's; OpenCV waits on the CPU instead
        std::lock_guard<std::mutex> lock(m_);
        [bufferLocked() encodeWaitForEvent:me->e_ value:me->value_];
        return true;
    }

    void synchronize() CV_OVERRIDE
    {
        id<MTLCommandBuffer> last = nil;
        {
            std::lock_guard<std::mutex> lock(m_);
            commitLocked();
            last = [last_ retain];      // waited on outside the lock, so other threads can queue
        }
        if (last)
        {
            [last waitUntilCompleted];
            [last release];
        }
        err_->check();
    }

    //! commits and waits, for the copies that have to finish before this call returns
    void flushAndWait()
    {
        id<MTLCommandBuffer> last = nil;
        {
            std::lock_guard<std::mutex> lock(m_);
            commitLocked();
            last = [last_ retain];
        }
        if (last)
        {
            [last waitUntilCompleted];
            [last release];
        }
        err_->check();
    }

    const int device_;
    id<MTLCommandQueue> q_;
    id<MTLSharedEvent> ev_;
    ErrorBoxPtr err_;
    std::mutex m_;

private:
    id<MTLCommandBuffer> cb_;       // open, not yet committed
    id<MTLCommandBuffer> last_;     // last committed, for synchronize()
    uint64_t counter_;
};

MetalStream& metalStreamOf(Stream& s)
{
    MetalStream* ms = dynamic_cast<MetalStream*>(&s);
    CV_Assert(ms && "accel(metal): the stream belongs to another backend");
    return *ms;
}

inline id<MTLBuffer> bufferOf(const DeviceRef& r)
{
    return (id<MTLBuffer>)r.handle;
}

/* How one UMat argument is addressed, 2-D or a 3-D batch of 2-D frames. A single image is one
   frame whose stride is never used. The same shape the HIP backend uses. */
struct Frames
{
    int n, rows, cols;
    size_t row, frame;      // bytes between rows, and between frames

    explicit Frames(const UMat& m)
    {
        const bool batched = m.dims == 3;
        n     = batched ? m.size[0] : 1;
        rows  = batched ? m.size[1] : m.rows;
        cols  = batched ? m.size[2] : m.cols;
        row   = m.step[m.dims - 2];
        frame = batched ? m.step[0] : 0;
    }
};

//! 2-D, or a 3-D batch; anything else the backend leaves to OpenCV
inline bool addressable(const UMat& m, int esz)
{
    return (m.dims == 2 || m.dims == 3) && kernels::supportedElementSize(esz);
}

/* A blit between buffers needs its offsets and lengths to be multiples of 4 on macOS. A UMat's
   offset is a multiple of its element size, which for CV_8UC3 is 3, and its step need not be
   aligned either, so the condition really does fail in practice. Where it does, the copy goes
   through the compute path instead, which has no such rule. */
inline bool blittable(size_t srcOff, size_t srcStep, size_t dstOff, size_t dstStep, size_t widthBytes)
{
    return ((srcOff | srcStep | dstOff | dstStep | widthBytes) & 3u) == 0;
}

enum KernelId { KERNEL_FLIP = 0, KERNEL_TRANSPOSE = 1 };

//! one Metal device and everything built lazily on top of it
struct Dev
{
    Dev() : dev(nil), lib(nil), utilityQueue(nil), unified(false),
            kinds(MEMORY_DEVICE), maxBuffer(0) {}

    id<MTLDevice> dev;
    id<MTLLibrary> lib;                 // the kernels, compiled once
    std::once_flag libOnce;
    std::string libError;               // why the compile failed, if it did

    id<MTLCommandQueue> utilityQueue;   // for the backend's own blits (Managed synchronization)
    std::once_flag utilityOnce;

    bool unified;                       // the CPU and the GPU share one pool of memory
    int kinds;                          // bitmask of MemoryKind
    size_t maxBuffer;                   // the largest buffer this device will make
    std::string name;

    // pipelines by (kernel, element size); building one compiles a specialization
    std::mutex pipeMutex;
    std::map<std::pair<int, int>, id<MTLComputePipelineState> > pipes;
};

class MetalBackend;

class MetalCoreOps CV_FINAL : public CoreOps
{
public:
    explicit MetalCoreOps(MetalBackend* b) : b_(b) {}

    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE;
    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE;

private:
    MetalBackend* b_;
};

class MetalBackend CV_FINAL : public Backend
{
public:
    MetalBackend() : core_(this)
    {
        NSArray<id<MTLDevice> >* found = nil;
#if TARGET_OS_OSX
        found = [MTLCopyAllDevices() autorelease];
#else
        id<MTLDevice> one = MTLCreateSystemDefaultDevice();
        if (one)
            found = [NSArray arrayWithObject:[one autorelease]];
#endif
        for (id<MTLDevice> d in found)
        {
            std::unique_ptr<Dev> e(new Dev());
            e->dev = [d retain];
            const char* n = [[d name] UTF8String];
            e->name = n ? n : "Metal device";

            /* Whether the two sides share memory. Before 10.15 there is no way to ask, and the
               parts that predate it with shared memory are Intel integrated GPUs, which report
               themselves as low power. */
            if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *))
                e->unified = d.hasUnifiedMemory;
            else
                e->unified = d.isLowPower;

            /* What this device can provide. Its own memory always. Where the two sides share a
               pool, a Shared buffer is reached by both with no copy at all and nothing has to
               be synchronized, so it is coherent. Where they do not, a Managed buffer is the
               pair of copies the non-coherent kind describes: the brackets below do the
               synchronizing. Managed is macOS only. */
            e->kinds = MEMORY_DEVICE;
            if (e->unified)
                e->kinds |= MEMORY_SHARED_COHERENT;
#if CV_METAL_HAS_MANAGED
            else
                e->kinds |= MEMORY_SHARED_NONCOHERENT;
#endif

            if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *))
                e->maxBuffer = (size_t)d.maxBufferLength;

            CV_LOG_INFO(NULL, "accel metal:" << devs_.size() << " " << e->name
                        << " unified=" << (int)e->unified
                        << " kinds=" << e->kinds
                        << " maxBuffer=" << e->maxBuffer);
            devs_.push_back(std::move(e));
        }
    }

    ~MetalBackend()
    {
        for (size_t i = 0; i < devs_.size(); i++)
        {
            Dev& d = *devs_[i];
            for (std::map<std::pair<int, int>, id<MTLComputePipelineState> >::iterator it = d.pipes.begin();
                 it != d.pipes.end(); ++it)
                [it->second release];
            [d.utilityQueue release];
            [d.lib release];
            [d.dev release];
        }
    }

    String name() const CV_OVERRIDE { return "metal"; }
    int deviceCount() const CV_OVERRIDE { return (int)devs_.size(); }
    String deviceDescription(int device) const CV_OVERRIDE { return devs_[device]->name; }
    int memoryKinds(int device) const CV_OVERRIDE { return devs_[device]->kinds; }

    /* On Apple silicon a Shared buffer costs nothing the CPU would otherwise avoid: it is the
       same DRAM a Private one would use, so making it reachable from the CPU is free and every
       map() that would have copied does not. On a discrete GPU the device's own memory is the
       only fast one, and a buffer the CPU touches is promoted later by the allocator. */
    MemoryKind defaultMemoryKind(int device) const CV_OVERRIDE
    {
        return devs_[device]->unified ? MEMORY_SHARED_COHERENT : MEMORY_DEVICE;
    }

    void* allocate(int device, size_t size, MemoryKind kind) CV_OVERRIDE
    {
        Dev& d = *devs_[device];
        CV_Assert((d.kinds & (int)kind) != 0 && "accel(metal): a kind this device does not provide");
        if (d.maxBuffer && size > d.maxBuffer)
        {
            // a 3-D batch can ask for more than the device will give in one buffer
            CV_LOG_WARNING(NULL, "accel(metal): " << size << " bytes exceeds the device's limit of "
                           << d.maxBuffer);
            return 0;
        }

        /* Untracked on purpose. OpenCV already orders every access through its own events, so
           Metal's automatic hazard tracking would re-derive the same dependencies and insert a
           second set of barriers for them. */
        MTLResourceOptions opt = MTLResourceHazardTrackingModeUntracked;
        switch (kind)
        {
        case MEMORY_SHARED_COHERENT:
            opt |= MTLResourceStorageModeShared;
            break;
#if CV_METAL_HAS_MANAGED
        case MEMORY_SHARED_NONCOHERENT:
            opt |= MTLResourceStorageModeManaged;
            break;
#endif
        default:
            opt |= MTLResourceStorageModePrivate;
            break;
        }

        id<MTLBuffer> b = [d.dev newBufferWithLength:(size ? size : 1) options:opt];
        return b;       // owned by the caller until deallocate(); nil on failure, which is NULL
    }

    void deallocate(int, void* ptr, MemoryKind) CV_OVERRIDE
    {
        // called while UMats are released: never throw
        [(id<MTLBuffer>)ptr release];
    }

    void* hostAddress(int, void* ptr, MemoryKind kind) CV_OVERRIDE
    {
        if (kind == MEMORY_DEVICE)
            return 0;       // Private memory has no CPU address at all
        return [(id<MTLBuffer>)ptr contents];
    }

    /* Managed memory is two copies and the GPU's is the one that was just written, so it has to
       be brought back before the CPU reads. Shared memory is one copy and needs nothing. */
    void beginCpuAccess(int device, void* ptr, size_t size, MemoryKind kind, bool /*write*/) CV_OVERRIDE
    {
#if CV_METAL_HAS_MANAGED
        if (kind != MEMORY_SHARED_NONCOHERENT || size == 0)
            return;
        Dev& d = *devs_[device];
        id<MTLCommandBuffer> cb = [utilityQueue(d) commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit synchronizeResource:(id<MTLBuffer>)ptr];
        [blit endEncoding];
        [cb commit];
        // the CPU reads these bytes as soon as this returns, so the copy has to be done
        [cb waitUntilCompleted];
        if (cb.error)
            CV_Error_(Error::GpuApiCallError,
                      ("accel(metal): synchronizing a managed buffer failed: %s", describe(cb.error).c_str()));
#else
        (void)device; (void)ptr; (void)size; (void)kind;
#endif
    }

    //! tells Metal which bytes the CPU changed, so the GPU's copy is refreshed from them
    void endCpuAccess(int device, void* ptr, size_t size, MemoryKind kind, bool written) CV_OVERRIDE
    {
#if CV_METAL_HAS_MANAGED
        if (kind != MEMORY_SHARED_NONCOHERENT || !written || size == 0)
            return;
        (void)device;
        [(id<MTLBuffer>)ptr didModifyRange:NSMakeRange(0, size)];
#else
        (void)device; (void)ptr; (void)size; (void)kind; (void)written;
#endif
    }

    /* Page-aligned rather than merely malloc'd. Metal can wrap page-aligned host memory as a
       buffer with newBufferWithBytesNoCopy:, so a host copy allocated here can later be handed
       to the GPU without being copied at all. Nothing does that yet. */
    void* allocateHost(int, size_t size) CV_OVERRIDE
    {
        void* p = 0;
        const size_t page = (size_t)getpagesize();
        const size_t rounded = ((size ? size : 1) + page - 1) / page * page;
        if (posix_memalign(&p, page, rounded) != 0)
            return 0;
        return p;
    }

    void deallocateHost(int, void* ptr) CV_OVERRIDE
    {
        free(ptr);
    }

    void copy(int device, CopyKind kind, DeviceRef dst, size_t dstStep,
              DeviceRef src, size_t srcStep, size_t widthBytes, size_t rows,
              Stream& stream, bool /*hostStable*/) CV_OVERRIDE
    {
        if (widthBytes == 0 || rows == 0)
            return;
        Dev& d = *devs_[device];
        MetalStream& s = metalStreamOf(stream);
        CV_Assert(s.device_ == device);

        switch (kind)
        {
        case COPY_DEVICE_TO_DEVICE:
            copy2d(s, d, bufferOf(dst), dst.offset, dstStep,
                   bufferOf(src), src.offset, srcStep, widthBytes, rows);
            break;

        case COPY_HOST_TO_DEVICE:
        {
            /* newBufferWithBytes: reads the host memory now, which is what hostStable false
               asks for: the caller may change those bytes the moment this returns. */
            const uchar* host = (const uchar*)src.handle + src.offset;
            id<MTLBuffer> staging = packHost(d, host, srcStep, widthBytes, rows);
            copy2d(s, d, bufferOf(dst), dst.offset, dstStep,
                   staging, 0, widthBytes, widthBytes, rows);
            releaseWhenDone(s, staging);
            break;
        }

        case COPY_DEVICE_TO_HOST:
        {
            id<MTLBuffer> staging = [d.dev newBufferWithLength:widthBytes * rows
                                                       options:MTLResourceStorageModeShared |
                                                               MTLResourceHazardTrackingModeUntracked];
            if (!staging)
                CV_Error(Error::GpuApiCallError, "accel(metal): could not allocate a staging buffer");
            copy2d(s, d, staging, 0, widthBytes,
                   bufferOf(src), src.offset, srcStep, widthBytes, rows);
            /* The bytes are read back here rather than in a completion handler: the interface
               lets OpenCV read the destination once it has waited for the stream, and a
               handler runs after the event is signalled, which would be too late. */
            s.flushAndWait();
            const uchar* from = (const uchar*)[staging contents];
            uchar* to = (uchar*)dst.handle + dst.offset;
            for (size_t r = 0; r < rows; r++)
                memcpy(to + r * dstStep, from + r * widthBytes, widthBytes);
            [staging release];
            break;
        }
        }
    }

    Ptr<Stream> createStream(int device) CV_OVERRIDE
    {
        return makePtr<MetalStream>(device, devs_[device]->dev);
    }

    Ops* ops(const String& module) CV_OVERRIDE
    {
        return module == CoreOps::module() ? &core_ : 0;
    }

    //! queues a kernel; called by MetalCoreOps
    void dispatch(MetalStream& s, int device, KernelId id, int esz,
                  id<MTLBuffer> srcBuf, id<MTLBuffer> dstBuf,
                  const kernels::Params& p, int batch)
    {
        Dev& d = *devs_[device];
        id<MTLComputePipelineState> pso = pipeline(d, id, esz);

        /* Threads per group shaped like the HIP backend's 32 x 8: the width is the hardware's
           own execution width, so a row of threads maps onto one SIMD group and neighbouring
           threads read neighbouring bytes. */
        NSUInteger tw = pso.threadExecutionWidth;
        NSUInteger th = pso.maxTotalThreadsPerThreadgroup / (tw ? tw : 1);
        if (th > 8)
            th = 8;
        if (th == 0)
            th = 1;

        const MTLSize group = MTLSizeMake(tw, th, 1);
        const MTLSize grid = MTLSizeMake((p.cols + tw - 1) / tw, (p.rows + th - 1) / th,
                                         (NSUInteger)batch);

        std::lock_guard<std::mutex> lock(s.m_);
        id<MTLComputeCommandEncoder> enc = [s.bufferLocked() computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:srcBuf offset:0 atIndex:kernels::ARG_SRC];
        [enc setBuffer:dstBuf offset:0 atIndex:kernels::ARG_DST];
        [enc setBytes:&p length:sizeof(p) atIndex:kernels::ARG_PARAMS];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [enc endEncoding];
    }

    Dev& device(int i) { return *devs_[i]; }

private:
    //! the queue the backend uses for its own work, apart from any stream
    id<MTLCommandQueue> utilityQueue(Dev& d)
    {
        std::call_once(d.utilityOnce, [&]() {
            d.utilityQueue = [d.dev newCommandQueue];
        });
        if (!d.utilityQueue)
            CV_Error(Error::GpuApiCallError, "accel(metal): could not create a command queue");
        return d.utilityQueue;
    }

    //! the kernels, compiled once per device
    id<MTLLibrary> library(Dev& d)
    {
        std::call_once(d.libOnce, [&]() {
            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:kernels::source()];
            MTLCompileOptions* opt = [[[MTLCompileOptions alloc] init] autorelease];
            d.lib = [d.dev newLibraryWithSource:src options:opt error:&err];
            if (!d.lib)
                d.libError = describe(err);
        });
        if (!d.lib)
            CV_Error_(Error::GpuApiCallError,
                      ("accel(metal): compiling the kernels failed: %s", d.libError.c_str()));
        return d.lib;
    }

    /* The pipeline for one kernel at one element size. The size is a function constant, so each
       size compiles its own specialization in which the byte copy is a fixed count of moves
       rather than a loop. Built on first use and kept for the life of the process. */
    id<MTLComputePipelineState> pipeline(Dev& d, KernelId id, int esz)
    {
        const std::pair<int, int> key((int)id, esz);
        {
            std::lock_guard<std::mutex> lock(d.pipeMutex);
            std::map<std::pair<int, int>, id<MTLComputePipelineState> >::iterator it = d.pipes.find(key);
            if (it != d.pipes.end())
                return it->second;
        }

        id<MTLLibrary> lib = library(d);
        NSError* err = nil;

        MTLFunctionConstantValues* consts = [[[MTLFunctionConstantValues alloc] init] autorelease];
        int value = esz;
        [consts setConstantValue:&value type:MTLDataTypeInt atIndex:kernels::FC_ELEMENT_SIZE];

        NSString* fname = id == KERNEL_FLIP ? @"accel_flip" : @"accel_transpose";
        id<MTLFunction> fn = [[lib newFunctionWithName:fname constantValues:consts error:&err] autorelease];
        if (!fn)
            CV_Error_(Error::GpuApiCallError,
                      ("accel(metal): specializing %s failed: %s", [fname UTF8String], describe(err).c_str()));

        id<MTLComputePipelineState> pso = [d.dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso)
            CV_Error_(Error::GpuApiCallError,
                      ("accel(metal): building a pipeline for %s failed: %s",
                       [fname UTF8String], describe(err).c_str()));

        std::lock_guard<std::mutex> lock(d.pipeMutex);
        // another thread may have built the same one meanwhile; keep whichever is already there
        std::pair<std::map<std::pair<int, int>, id<MTLComputePipelineState> >::iterator, bool> ins =
            d.pipes.insert(std::make_pair(key, pso));
        if (!ins.second)
        {
            [pso release];
            return ins.first->second;
        }
        return pso;
    }

    /* A rectangle of host memory as one tightly packed Shared buffer. The rows are packed here
       so that the copy onto the device is a single run of bytes whenever its own step allows. */
    id<MTLBuffer> packHost(Dev& d, const uchar* host, size_t srcStep, size_t widthBytes, size_t rows)
    {
        const size_t total = widthBytes * rows;
        const MTLResourceOptions opt = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> staging = nil;
        if (srcStep == widthBytes)
        {
            staging = [d.dev newBufferWithBytes:host length:total options:opt];
        }
        else
        {
            staging = [d.dev newBufferWithLength:total options:opt];
            if (staging)
            {
                uchar* to = (uchar*)[staging contents];
                for (size_t r = 0; r < rows; r++)
                    memcpy(to + r * widthBytes, host + r * srcStep, widthBytes);
            }
        }
        if (!staging)
            CV_Error(Error::GpuApiCallError, "accel(metal): could not allocate a staging buffer");
        return staging;
    }

    //! drops the backend's reference to @p b once the work using it has finished
    void releaseWhenDone(MetalStream& s, id<MTLBuffer> b)
    {
        std::lock_guard<std::mutex> lock(s.m_);
        [s.bufferLocked() addCompletedHandler:^(id<MTLCommandBuffer>) { [b release]; }];
    }

    /* Rows of bytes from one buffer to another. A blit is the direct way and is used whenever
       Metal's alignment rule allows it; otherwise the flip kernel does it, with one byte per
       element and neither axis flipped, which is a plain 2-D copy. */
    void copy2d(MetalStream& s, Dev& d,
                id<MTLBuffer> dstBuf, size_t dstOff, size_t dstStep,
                id<MTLBuffer> srcBuf, size_t srcOff, size_t srcStep,
                size_t widthBytes, size_t rows)
    {
        const bool contiguous = srcStep == widthBytes && dstStep == widthBytes;
        if (blittable(srcOff, srcStep, dstOff, dstStep, widthBytes))
        {
            std::lock_guard<std::mutex> lock(s.m_);
            id<MTLBlitCommandEncoder> blit = [s.bufferLocked() blitCommandEncoder];
            if (contiguous)
            {
                [blit copyFromBuffer:srcBuf sourceOffset:srcOff
                            toBuffer:dstBuf destinationOffset:dstOff
                                size:widthBytes * rows];
            }
            else
            {
                for (size_t r = 0; r < rows; r++)
                    [blit copyFromBuffer:srcBuf sourceOffset:srcOff + r * srcStep
                                toBuffer:dstBuf destinationOffset:dstOff + r * dstStep
                                    size:widthBytes];
            }
            [blit endEncoding];
            return;
        }

        kernels::Params p;
        memset(&p, 0, sizeof(p));
        p.soffset = srcOff;
        p.doffset = dstOff;
        p.sstep = srcStep;
        p.dstep = dstStep;
        p.rows = (uint32_t)rows;
        p.cols = (uint32_t)widthBytes;      // one byte per element, so a column is a byte
        dispatch(s, s.device_, KERNEL_FLIP, 1, srcBuf, dstBuf, p, 1);
    }

    std::vector<std::unique_ptr<Dev> > devs_;
    MetalCoreOps core_;
};

bool MetalCoreOps::flip(const UMat& src, UMat& dst, int flipCode, Stream& stream)
{
    const int esz = (int)src.elemSize();
    if (!addressable(src, esz) || src.dims != dst.dims)
        return false;

    const Frames s_(src), d_(dst);
    MetalStream& s = metalStreamOf(stream);
    const DeviceRef sr = deviceRef(src), dr = deviceRef(dst);

    kernels::Params p;
    memset(&p, 0, sizeof(p));
    p.soffset = sr.offset;
    p.doffset = dr.offset;
    p.sstep = s_.row;
    p.sframe = s_.frame;
    p.dstep = d_.row;
    p.dframe = d_.frame;
    p.rows = (uint32_t)s_.rows;
    p.cols = (uint32_t)s_.cols;
    // OpenCV's codes: 0 around the x axis (rows), > 0 around the y axis (columns), < 0 both
    p.flipRows = flipCode <= 0;
    p.flipCols = flipCode != 0;

    b_->dispatch(s, s.device_, KERNEL_FLIP, esz, bufferOf(sr), bufferOf(dr), p, s_.n);
    return true;
}

bool MetalCoreOps::transpose(const UMat& src, UMat& dst, Stream& stream)
{
    const int esz = (int)src.elemSize();
    if (!addressable(src, esz) || src.dims != dst.dims)
        return false;

    const Frames s_(src), d_(dst);
    MetalStream& s = metalStreamOf(stream);
    const DeviceRef sr = deviceRef(src), dr = deviceRef(dst);

    kernels::Params p;
    memset(&p, 0, sizeof(p));
    p.soffset = sr.offset;
    p.doffset = dr.offset;
    p.sstep = s_.row;
    p.sframe = s_.frame;
    p.dstep = d_.row;
    p.dframe = d_.frame;
    p.rows = (uint32_t)s_.rows;
    p.cols = (uint32_t)s_.cols;

    b_->dispatch(s, s.device_, KERNEL_TRANSPOSE, esz, bufferOf(sr), bufferOf(dr), p, s_.n);
    return true;
}

} // namespace

#ifndef BUILD_PLUGIN
//! built into core: the registry creates it (registry.cpp), there is no plugin to load
Ptr<Backend> createMetalBackend()
{
    return makePtr<MetalBackend>();
}
#endif

}}} // namespace cv::accel::metal_impl

#ifdef BUILD_PLUGIN
CV_ACCEL_DEFINE_PLUGIN(new cv::accel::metal_impl::MetalBackend())
#endif

#endif // HAVE_ACCEL_METAL
