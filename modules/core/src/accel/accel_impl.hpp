// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_IMPL_HPP
#define OPENCV_CORE_SRC_ACCEL_IMPL_HPP

#include "opencv2/core/accel_backend.hpp"
#include "opencv2/core/utils/accel_dispatch.private.hpp"

#include <atomic>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace cv { namespace accel {

namespace detail { class DeviceAllocator; }

#ifdef HAVE_CUDA
namespace impl {
//! the built-in CUDA backend (backend_cuda.cpp)
Ptr<Backend> createCudaBackend();
}
#endif

#if defined(HAVE_ACCEL_HIP) && !defined(BUILD_PLUGIN)
namespace hip_impl {
//! the HIP backend (backend_hip.cpp) when it is built into core rather than as a plugin
Ptr<Backend> createHipBackend();
}
#endif

struct Device::Impl
{
    Impl(const Ptr<Backend>& b, int i);

    Ptr<Backend> backend;
    int index;
    String name;                        // "cuda:0"
    int kinds;                          // backend->memoryKinds(index)
    MemoryKind defaultKind;             // backend->defaultMemoryKind(index)
    std::atomic<size_t> pinnedBytes;    // allocated with a shared kind, against the budget
    detail::DeviceAllocator* allocator; // lives as long as the process

    /* Released memory kept for the next buffer of the same size and kind, because asking the
       driver again is dear: measured on ROCm, an allocation from the stream-ordered pool costs
       about 15 us whatever its size, and pinning host memory about 0.2 ms per megabyte. A
       pipeline that makes a buffer per frame pays that on every frame. Guarded by its own
       mutex, bounded by OPENCV_ACCEL_CACHE_MAX_MB, and emptied when the device runs out. */
    std::mutex cacheMutex;
    std::map<std::pair<size_t, int>, std::vector<void*> > cache;
    size_t cachedBytes;                 // under cacheMutex

    //! true when the device has a kind the CPU addresses, whatever a given buffer uses
    bool canShare() const { return (kinds & ~(int)MEMORY_DEVICE) != 0; }

    //! the stream cv:: calls use on this device outside an ExecutionScope with a stream of its own
    const Ptr<Stream>& defaultStream();

private:
    std::mutex streamMutex_;
    Ptr<Stream> stream_;
};

namespace detail {

/*
State of one device buffer, kept in UMatData::allocatorContext.

The UMatData flags say which copy is current, as in the T-API: HOST_COPY_OBSOLETE and
DEVICE_COPY_OBSOLETE for memory the CPU reaches through a copy (COPY_ON_MAP is set). Shared
memory has one copy only; there the flags are unused.

The events order device work against the CPU and against other streams: CPU reads wait for
the last write, CPU writes and the release of the memory wait for the reads as well. Streams
are held, so a stream is never confused with a later one at the same address. All fields are
protected by the UMatData lock.

Shared memory can be reached by other devices through a dma-buf (setPreferredDevices). Such
an import is an alias: a UMatData of the other device over the same bytes, owned by this
buffer and released with it. Device work on an alias is ordered by the events of the buffer it
imports, so one set of events covers every device touching those bytes.
*/
struct BufferState
{
    explicit BufferState(Device::Impl* d)
        : dev(d), kind(d->defaultKind), kindRequested(false), initialized(false), mapped(false),
          mappedWrite(false), imported(false), externallyReferenced(false), aliasOf(0) {}

    Device::Impl* dev;
    MemoryKind kind;        // the kind this buffer's memory was allocated with
    /* The kind was asked for, by UMatUsageFlags, by the debug override, or by the pinned
       budget turning a shared choice down. Such a buffer is never moved to another kind
       later: the caller said what it wanted, and the budget means it cannot be afforded. */
    bool kindRequested;
    bool initialized;       // something was written into it; until then there is nothing to download
    bool mapped;            // Mat views exist (UMat::getMat)
    bool mappedWrite;       // ... and one of them may write
    bool imported;          // memory of someone else (wrapDmaBuf)
    /* Someone outside OpenCV holds a reference to these exact bytes: a device pointer from
       UMat::handle(), or a dma-buf exported with exportDmaBuf(). Both are handed out without a
       way to learn when they are done with, so this is never cleared and the memory never moves. */
    bool externallyReferenced;
    Ptr<Event> write;       // the last device work writing the buffer
    Ptr<Stream> writeStream;
    std::vector<std::pair<Ptr<Stream>, Ptr<Event> > > reads;   // last read per stream since that write

    UMatData* aliasOf;                  // in an alias: the buffer whose bytes it imports
    std::vector<UMatData*> aliases;     // imports of these bytes on other devices
    std::vector<Device::Impl*> refused; // devices that cannot import them
};

/*
MatAllocator of one device. UMat data it creates lives in the device's memory, and the CPU
reaches it the way the T-API does: map() when the first Mat view is taken, unmap() when the
last one goes. For copy-only memory map() downloads into a host copy when that is stale and
unmap() uploads what the CPU wrote; for shared memory both only order the CPU access against
device work and keep the CPU caches right.
*/
class DeviceAllocator CV_FINAL : public MatAllocator
{
public:
    explicit DeviceAllocator(Device::Impl* dev) : dev_(dev) {}

    UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step,
                       AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE;
    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE;
    void deallocate(UMatData* u) const CV_OVERRIDE;
    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE;
    void unmap(UMatData* u) const CV_OVERRIDE;
    void download(UMatData* u, void* dst, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const CV_OVERRIDE;
    void upload(UMatData* u, const void* src, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE;
    void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool sync) const CV_OVERRIDE;

    //! a buffer over an imported dma-buf of @p size bytes; NULL when the device cannot import it
    UMatData* importDmaBuf(int fd, size_t size) const;

    Device::Impl* device() const { return dev_; }

private:
    Device::Impl* dev_;
};

/** @brief The kind a new buffer on @p dev will get, before its size or its UMatUsageFlags are
known: the debug override when it applies, otherwise the device's default. For predicting what
a buffer that does not exist yet will cost. */
MemoryKind newBufferKind(Device::Impl* dev);

//! the state of device memory; NULL for host memory, OpenCL buffers and other UMats
BufferState* stateOf(const UMatData* u);
bool isDeviceAllocator(const MatAllocator* a);
//! the buffer whose events order work on @p u: @p u itself, or for an alias the buffer it imports
UMatData* owningData(UMatData* u);

//! the CPU waits for the device work its access conflicts with: writes, and reads too for @p write
void waitOnCpu(BufferState* st, bool write);
//! makes the work queued next on @p s wait for the device work it conflicts with
void orderOnStream(BufferState* st, const Ptr<Stream>& s, bool write);
//! records that the work queued on @p s up to @p e reads, or for @p write writes, the buffer
void recordUse(BufferState* st, const Ptr<Stream>& s, const Ptr<Event>& e, bool write);
//! UMat::handle(): the CPU waits for device work, so the caller's own work sees finished data
void prepareHandle(UMatData* u, AccessFlag accessFlags);

/** @brief An import of the bytes of shared memory @p u on device @p dev, made through a dma-buf
on first use and kept with @p u. Returns @p u for its own device, NULL when the bytes cannot be
shared with @p dev. */
UMatData* aliasOn(UMatData* u, Device::Impl* dev);

/** @brief Frees released device memory whose device work has finished.
Memory released while device work still used it waits in a list; with @p wait the CPU
waits for that work and frees everything. */
void collectRetired(bool wait);

//! the stream for work on @p dev: this thread's ExecutionScope stream there, otherwise the device's default stream
const Ptr<Stream>& currentStream(Device::Impl* dev);
//! for UMat::create: the allocator of this thread's device ExecutionScope, NULL otherwise
MatAllocator* scopeAllocator();
//! the device of this thread's ExecutionScope (NULL for a CPU scope); @p active tells whether a scope exists
Device::Impl* scopeDevice(bool& active);
//! setPreferredDevices(), in order
std::vector<Device::Impl*> preferredDevices();

void countCopy(CopyKind kind, size_t bytes);
void countDeviceOp();
void countFallback();
void countHostWait();

//! device buffers alive in the process (they make isActive() true)
extern std::atomic<int> g_liveBuffers;

}}} // namespace cv::accel::detail

#endif // OPENCV_CORE_SRC_ACCEL_IMPL_HPP
