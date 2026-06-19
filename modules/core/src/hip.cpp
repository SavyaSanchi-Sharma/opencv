#define OPENCV_CORE_HIP_IMPL
#include "precomp.hpp"
#include "opencv2/core/hip.hpp"
#include "opencv2/core/hip_stream_accessor.hpp"
#include "opencv2/core/private/hip_stubs.hpp"
#include "opencv2/core/utils/allocator_stats.impl.hpp"
#include "umatrix.hpp"
#include <cstdint>
#include <string>
#include <sstream>

#ifdef HAVE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

using namespace cv;
using namespace cv::hip;

#ifdef HAVE_HIP

namespace cv { namespace hip {
static inline void checkHipError(hipError_t err, const char* file, int line, const char* func) {
    if (err != hipSuccess)
        cv::error(cv::Error::GpuApiCallError, hipGetErrorString(err), func, file, line);
}
}}

#define hipSafeCall(expr) cv::hip::checkHipError((expr), __FILE__, __LINE__, CV_Func)

namespace cv { namespace hip { namespace device {
    void copyToWithMask(const HipMat& src, HipMat& dst, const HipMat& mask, Stream& stream);
    void setToWithoutMask(HipMat& mat, Scalar val, Stream& stream);
    void setToWithMask(HipMat& mat, const HipMat& mask, Scalar val, Stream& stream);
    void convertToNoScale(const HipMat& src, HipMat& dst, Stream& stream);
    void convertToScale(const HipMat& src, HipMat& dst, double alpha, double beta, Stream& stream);
}}}

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
        if (!srcdata || !dstdata || !srcdata->handle || !dstdata->handle) return;
        if (dims <= 2) {
            const uchar* src = (const uchar*)srcdata->handle;
            uchar*       dst = (uchar*)dstdata->handle;
            if (dims == 2) {
                src += srcofs[0] * srcstep[0] + srcofs[1];
                dst += dstofs[0] * dststep[0] + dstofs[1];
            }
            if (sync)
                hipSafeCall(hipMemcpy2D(dst, dststep[0], src, srcstep[0],
                                        sz[dims - 1], sz[0],
                                        hipMemcpyDeviceToDevice));
            else
                hipSafeCall(hipMemcpy2DAsync(dst, dststep[0], src, srcstep[0],
                                             sz[dims - 1], sz[0],
                                             hipMemcpyDeviceToDevice, 0));
        } else {
            hipSafeCall(hipMemcpy(dstdata->handle, srcdata->handle,
                                  srcdata->size, hipMemcpyDeviceToDevice));
        }
        dstdata->markHostCopyObsolete(true);
        dstdata->markDeviceCopyObsolete(false);
    }
};

HipAllocator hipAllocatorInstance;

} // anonymous

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

// HipMat::Allocator statics — default is nullptr; create() routes through HipAllocator

#ifdef HAVE_HIP
namespace {
HipMat::Allocator* g_defaultAllocator = nullptr;
HipMat::Allocator* g_stdAllocator     = nullptr;
} // anonymous
#endif

// ======================== HipMat::Allocator statics ========================

HipMat::Allocator* cv::hip::HipMat::defaultAllocator()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    return g_defaultAllocator;
#endif
}

void cv::hip::HipMat::setDefaultAllocator(HipMat::Allocator* allocator_)
{
#ifndef HAVE_HIP
    CV_UNUSED(allocator_); throw_no_hip();
#else
    CV_Assert(allocator_ != nullptr);
    g_defaultAllocator = allocator_;
#endif
}

HipMat::Allocator* cv::hip::HipMat::getStdAllocator()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    return g_stdAllocator;
#endif
}

// ======================== HipMat constructors ========================

cv::hip::HipMat::HipMat(HipMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      allocator(allocator_), u(nullptr)
{}

cv::hip::HipMat::HipMat(int rows_, int cols_, int type_, HipMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      allocator(allocator_), u(nullptr)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

cv::hip::HipMat::HipMat(Size size_, int type_, HipMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      allocator(allocator_), u(nullptr)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

cv::hip::HipMat::HipMat(int rows_, int cols_, int type_, Scalar s_, HipMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      allocator(allocator_), u(nullptr)
{
    if (rows_ > 0 && cols_ > 0) {
        create(rows_, cols_, type_);
        setTo(s_);
    }
}

cv::hip::HipMat::HipMat(Size size_, int type_, Scalar s_, HipMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      allocator(allocator_), u(nullptr)
{
    if (size_.height > 0 && size_.width > 0) {
        create(size_.height, size_.width, type_);
        setTo(s_);
    }
}

cv::hip::HipMat::HipMat(const HipMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step),
      data(m.data), refcount(m.refcount),
      datastart(m.datastart), dataend(m.dataend),
      allocator(m.allocator), u(m.u)
{
    if (refcount) CV_XADD(refcount, 1);
}

cv::hip::HipMat::HipMat(int rows_, int cols_, int type_, void* data_, size_t step_)
    : flags(Mat::MAGIC_VAL | CV_MAT_CONT_FLAG | type_),
      rows(rows_), cols(cols_),
      step(step_ == Mat::AUTO_STEP ? (size_t)(cols_ * CV_ELEM_SIZE(type_)) : step_),
      data((uchar*)data_), refcount(nullptr),
      datastart((uchar*)data_),
      dataend((uchar*)data_ + step * rows_),
      allocator(nullptr), u(nullptr)
{
    updateContinuityFlag();
}

cv::hip::HipMat::HipMat(Size size_, int type_, void* data_, size_t step_)
    : HipMat(size_.height, size_.width, type_, data_, step_)
{}

cv::hip::HipMat::HipMat(const HipMat& m, Range rowRange_, Range colRange_)
    : flags(m.flags), rows(0), cols(0), step(m.step),
      data(m.data), refcount(m.refcount),
      datastart(m.datastart), dataend(m.dataend),
      allocator(m.allocator), u(m.u)
{
    if (rowRange_ == Range::all()) rowRange_ = Range(0, m.rows);
    if (colRange_ == Range::all()) colRange_ = Range(0, m.cols);
    CV_Assert(0 <= rowRange_.start && rowRange_.end <= m.rows);
    CV_Assert(0 <= colRange_.start && colRange_.end <= m.cols);
    data     += rowRange_.start * step + colRange_.start * m.elemSize();
    rows      = rowRange_.size();
    cols      = colRange_.size();
    dataend   = data + step * (rows - 1) + cols * m.elemSize();
    if (refcount) CV_XADD(refcount, 1);
    updateContinuityFlag();
}

cv::hip::HipMat::HipMat(const HipMat& m, Rect roi)
    : HipMat(m, Range(roi.y, roi.y + roi.height), Range(roi.x, roi.x + roi.width))
{}

cv::hip::HipMat::HipMat(InputArray arr, HipMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      allocator(allocator_), u(nullptr)
{
    upload(arr);
}

cv::hip::HipMat::~HipMat() { release(); }

HipMat& cv::hip::HipMat::operator=(const HipMat& m)
{
    if (this != &m) { HipMat tmp(m); swap(tmp); }
    return *this;
}

// ======================== create / release / fit / swap ========================

void cv::hip::HipMat::updateContinuityFlag()
{
    int sz[]     = { rows, cols };
    size_t st[]  = { step, elemSize() };
    flags = cv::updateContinuityFlag(flags, 2, sz, st);
}

void cv::hip::HipMat::create(int rows_, int cols_, int type_)
{
#ifndef HAVE_HIP
    CV_UNUSED(rows_); CV_UNUSED(cols_); CV_UNUSED(type_); throw_no_hip();
#else
    type_ = CV_MAT_TYPE(type_);
    if (rows == rows_ && cols == cols_ && type() == type_ && data) return;
    if (data) release();
    CV_DbgAssert(rows_ >= 0 && cols_ >= 0);
    if (rows_ == 0 || cols_ == 0) return;

    flags = Mat::MAGIC_VAL | type_;
    rows  = rows_;
    cols  = cols_;

    int sizes[2] = { rows_, cols_ };
    u = hipAllocatorInstance.allocate(2, sizes, type_, nullptr, &step,
                                      ACCESS_READ | ACCESS_WRITE, USAGE_DEFAULT);
    data      = datastart = (uchar*)u->handle;
    dataend   = data + step * rows_;
    refcount  = (int*)fastMalloc(sizeof(int));
    *refcount = 1;
    updateContinuityFlag();
#endif
}

void cv::hip::HipMat::create(Size size_, int type_) { create(size_.height, size_.width, type_); }

void cv::hip::HipMat::release()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (refcount && CV_XADD(refcount, -1) == 1) {
        if (u) {
            hipAllocatorInstance.deallocate(u);
            u = nullptr;
        }
        fastFree(refcount);
    }
    data = datastart = nullptr;
    dataend  = nullptr;
    step     = 0;
    rows = cols = 0;
    flags    = 0;
    refcount = nullptr;
    allocator = nullptr;
    u = nullptr;
#endif
}

void cv::hip::HipMat::fit(int rows_, int cols_, int type_)
{
    type_ = CV_MAT_TYPE(type_);
    if (rows == rows_ && cols == cols_ && type() == type_ && data) return;
    if (data && refcount && *refcount == 1 && u) {
        hipAllocatorInstance.deallocate(u);
        u = nullptr;
        data = datastart = nullptr;
        dataend = nullptr;
        fastFree(refcount);
        refcount = nullptr;
    }
    create(rows_, cols_, type_);
}

void cv::hip::HipMat::fit(Size size_, int type_) { fit(size_.height, size_.width, type_); }

void cv::hip::HipMat::swap(HipMat& mat)
{
    std::swap(flags,     mat.flags);
    std::swap(rows,      mat.rows);
    std::swap(cols,      mat.cols);
    std::swap(step,      mat.step);
    std::swap(data,      mat.data);
    std::swap(refcount,  mat.refcount);
    std::swap(datastart, mat.datastart);
    std::swap(dataend,   mat.dataend);
    std::swap(allocator, mat.allocator);
    std::swap(u,         mat.u);
}

// ======================== upload / download ========================

void cv::hip::HipMat::upload(InputArray arr)
{
#ifndef HAVE_HIP
    CV_UNUSED(arr); throw_no_hip();
#else
    Mat mat = arr.getMat();
    CV_DbgAssert(!mat.empty());
    create(mat.size(), mat.type());
    hipSafeCall(hipMemcpy2D(data, step,
                             mat.data, mat.step,
                             cols * elemSize(), rows,
                             hipMemcpyHostToDevice));
#endif
}

void cv::hip::HipMat::upload(InputArray arr, Stream& stream_)
{
#ifndef HAVE_HIP
    CV_UNUSED(arr); CV_UNUSED(stream_); throw_no_hip();
#else
    Mat mat = arr.getMat();
    CV_DbgAssert(!mat.empty());
    create(mat.size(), mat.type());
    hipStream_t s = StreamAccessor::getStream(stream_);
    hipSafeCall(hipMemcpy2DAsync(data, step,
                                  mat.data, mat.step,
                                  cols * elemSize(), rows,
                                  hipMemcpyHostToDevice, s));
#endif
}

void cv::hip::HipMat::download(OutputArray dst_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    dst_.create(size(), type());
    Mat dst = dst_.getMat();
    size_t widthBytes = cols * elemSize();
    size_t dstStep    = rows > 1 ? (size_t)dst.step : widthBytes;
    hipSafeCall(hipMemcpy2D(dst.data, dstStep,
                             data, step,
                             widthBytes, rows,
                             hipMemcpyDeviceToHost));
#endif
}

void cv::hip::HipMat::download(OutputArray dst_, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst_); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    dst_.create(size(), type());
    Mat dst = dst_.getMat();
    hipStream_t s = StreamAccessor::getStream(stream_);
    hipSafeCall(hipMemcpy2DAsync(dst.data, dst.step,
                                  data, step,
                                  cols * elemSize(), rows,
                                  hipMemcpyDeviceToHost, s));
#endif
}

// ======================== copyTo ========================

void cv::hip::HipMat::copyTo(OutputArray dst_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    dst_.create(size(), type());
    if (dst_.isUMat())
    {
        UMat dstUMat = dst_.getUMat(ACCESS_WRITE);
        if (dstUMat.u && dstUMat.u->currAllocator == getHipAllocator())
        {
            hipSafeCall(hipMemcpy2D(dstUMat.u->handle, dstUMat.step[0],
                                     data, step,
                                     cols * elemSize(), rows,
                                     hipMemcpyDeviceToDevice));
            return;
        }
    }
    Mat dst = dst_.getMat(ACCESS_WRITE);
    hipSafeCall(hipMemcpy2D(dst.data, dst.step,
                             data, step,
                             cols * elemSize(), rows,
                             hipMemcpyDeviceToHost));
#endif
}

void cv::hip::HipMat::copyTo(OutputArray dst_, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst_); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    dst_.create(size(), type());
    hipStream_t s = StreamAccessor::getStream(stream_);
    if (dst_.isUMat())
    {
        UMat dstUMat = dst_.getUMat(ACCESS_WRITE);
        if (dstUMat.u && dstUMat.u->currAllocator == getHipAllocator())
        {
            hipSafeCall(hipMemcpy2DAsync(dstUMat.u->handle, dstUMat.step[0],
                                          data, step,
                                          cols * elemSize(), rows,
                                          hipMemcpyDeviceToDevice, s));
            return;
        }
    }
    Mat dst = dst_.getMat(ACCESS_WRITE);
    hipSafeCall(hipMemcpy2DAsync(dst.data, dst.step,
                                  data, step,
                                  cols * elemSize(), rows,
                                  hipMemcpyDeviceToHost, s));
#endif
}

void cv::hip::HipMat::copyTo(OutputArray dst_, InputArray mask_) const
{
    copyTo(dst_, mask_, Stream::Null());
}

void cv::hip::HipMat::copyTo(OutputArray dst_, InputArray mask_, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst_); CV_UNUSED(mask_); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    dst_.create(size(), type());
    CV_Assert(dst_.isUMat() && mask_.isUMat());
    UMat dstUMat  = dst_.getUMat(ACCESS_WRITE);
    UMat maskUMat = mask_.getUMat();
    CV_Assert(dstUMat.u  && dstUMat.u->currAllocator  == getHipAllocator());
    CV_Assert(maskUMat.u && maskUMat.u->currAllocator == getHipAllocator());
    HipMat dst (dstUMat.rows,  dstUMat.cols,  dstUMat.type(),  dstUMat.u->handle,  dstUMat.step[0]);
    HipMat mask(maskUMat.rows, maskUMat.cols, maskUMat.type(), maskUMat.u->handle, maskUMat.step[0]);
    device::copyToWithMask(*this, dst, mask, stream_);
#endif
}

// ======================== setTo ========================

HipMat& cv::hip::HipMat::setTo(Scalar s) { return setTo(s, Stream::Null()); }

HipMat& cv::hip::HipMat::setTo(Scalar s, Stream& stream_)
{
#ifndef HAVE_HIP
    CV_UNUSED(s); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    device::setToWithoutMask(*this, s, stream_);
    return *this;
#endif
}

HipMat& cv::hip::HipMat::setTo(Scalar s, InputArray mask_) { return setTo(s, mask_, Stream::Null()); }

HipMat& cv::hip::HipMat::setTo(Scalar s, InputArray mask_, Stream& stream_)
{
#ifndef HAVE_HIP
    CV_UNUSED(s); CV_UNUSED(mask_); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    CV_Assert(mask_.isUMat());
    UMat maskUMat = mask_.getUMat();
    CV_Assert(maskUMat.u && maskUMat.u->currAllocator == getHipAllocator());
    HipMat mask(maskUMat.rows, maskUMat.cols, maskUMat.type(), maskUMat.u->handle, maskUMat.step[0]);
    device::setToWithMask(*this, mask, s, stream_);
    return *this;
#endif
}

// ======================== convertTo ========================

void cv::hip::HipMat::convertTo(OutputArray dst_, int rtype) const { convertTo(dst_, rtype, Stream::Null()); }

void cv::hip::HipMat::convertTo(OutputArray dst_, int rtype, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst_); CV_UNUSED(rtype); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    if (rtype < 0) rtype = type();
    else rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());
    dst_.create(size(), rtype);
    CV_Assert(dst_.isUMat());
    UMat dstUMat = dst_.getUMat(ACCESS_WRITE);
    CV_Assert(dstUMat.u && dstUMat.u->currAllocator == getHipAllocator());
    HipMat dst(dstUMat.rows, dstUMat.cols, dstUMat.type(), dstUMat.u->handle, dstUMat.step[0]);
    device::convertToNoScale(*this, dst, stream_);
#endif
}

void cv::hip::HipMat::convertTo(OutputArray dst_, int rtype, double alpha, double beta) const
{
    convertTo(dst_, rtype, alpha, beta, Stream::Null());
}

void cv::hip::HipMat::convertTo(OutputArray dst_, int rtype, double alpha, Stream& stream_) const
{
    convertTo(dst_, rtype, alpha, 0.0, stream_);
}

void cv::hip::HipMat::convertTo(OutputArray dst_, int rtype, double alpha, double beta, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst_); CV_UNUSED(rtype); CV_UNUSED(alpha); CV_UNUSED(beta); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    if (rtype < 0) rtype = type();
    else rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());
    dst_.create(size(), rtype);
    CV_Assert(dst_.isUMat());
    UMat dstUMat = dst_.getUMat(ACCESS_WRITE);
    CV_Assert(dstUMat.u && dstUMat.u->currAllocator == getHipAllocator());
    HipMat dst(dstUMat.rows, dstUMat.cols, dstUMat.type(), dstUMat.u->handle, dstUMat.step[0]);
    device::convertToScale(*this, dst, alpha, beta, stream_);
#endif
}

// ======================== HipMat& convenience overloads ========================

void cv::hip::HipMat::copyTo(HipMat& dst) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    dst.create(rows, cols, type());
    hipSafeCall(hipMemcpy2D(dst.data, dst.step, data, step,
                             cols * elemSize(), rows, hipMemcpyDeviceToDevice));
#endif
}

void cv::hip::HipMat::copyTo(HipMat& dst, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    dst.create(rows, cols, type());
    hipStream_t s = StreamAccessor::getStream(stream_);
    hipSafeCall(hipMemcpy2DAsync(dst.data, dst.step, data, step,
                                  cols * elemSize(), rows, hipMemcpyDeviceToDevice, s));
#endif
}

void cv::hip::HipMat::copyTo(HipMat& dst, InputArray mask_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst); CV_UNUSED(mask_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    uchar* old_data = dst.data;
    dst.create(rows, cols, type());
    if (dst.data != old_data)
        hipSafeCall(hipMemset2D(dst.data, dst.step, 0, cols * elemSize(), rows));
    CV_Assert(mask_.isUMat());
    UMat maskUMat = mask_.getUMat();
    CV_Assert(maskUMat.u && maskUMat.u->currAllocator == getHipAllocator());
    HipMat mask(maskUMat.rows, maskUMat.cols, maskUMat.type(), maskUMat.u->handle, maskUMat.step[0]);
    device::copyToWithMask(*this, dst, mask, Stream::Null());
#endif
}

void cv::hip::HipMat::copyTo(HipMat& dst, HipMat& mask_, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst); CV_UNUSED(mask_); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    uchar* old_data = dst.data;
    dst.create(rows, cols, type());
    if (dst.data != old_data) {
        hipStream_t s = StreamAccessor::getStream(stream_);
        hipSafeCall(hipMemset2DAsync(dst.data, dst.step, 0, cols * elemSize(), rows, s));
    }
    device::copyToWithMask(*this, dst, mask_, stream_);
#endif
}

void cv::hip::HipMat::convertTo(HipMat& dst, int rtype) const { convertTo(dst, rtype, Stream::Null()); }

void cv::hip::HipMat::convertTo(HipMat& dst, int rtype, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst); CV_UNUSED(rtype); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    if (rtype < 0) rtype = type();
    else rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());
    dst.create(rows, cols, rtype);
    device::convertToNoScale(*this, dst, stream_);
#endif
}

void cv::hip::HipMat::convertTo(HipMat& dst, int rtype, double alpha, double beta, Stream& stream_) const
{
#ifndef HAVE_HIP
    CV_UNUSED(dst); CV_UNUSED(rtype); CV_UNUSED(alpha); CV_UNUSED(beta); CV_UNUSED(stream_); throw_no_hip();
#else
    CV_DbgAssert(!empty());
    if (rtype < 0) rtype = type();
    else rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());
    dst.create(rows, cols, rtype);
    if (alpha == 1 && beta == 0)
        device::convertToNoScale(*this, dst, stream_);
    else
        device::convertToScale(*this, dst, alpha, beta, stream_);
#endif
}

void cv::hip::HipMat::assignTo(HipMat& m, int type_) const
{
    if (type_ < 0) m = *this;
    else convertTo(m, type_);
}

// ======================== Shape / ROI ========================

HipMat cv::hip::HipMat::reshape(int new_cn, int new_rows) const
{
    HipMat hdr = *this;
    if (new_cn == 0) new_cn = channels();
    size_t total_width = (size_t)cols * ((size_t)elemSize() / elemSize1());
    CV_Assert((total_width & (new_cn - 1)) == 0);
    int new_cols = (int)(total_width / new_cn);
    if (new_rows == 0) {
        new_rows = rows * cols / new_cols;
        CV_Assert(new_rows * new_cols == rows * cols);
    }
    if (new_rows != rows)
        CV_Assert(isContinuous());
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);
    hdr.rows  = new_rows;
    hdr.cols  = new_cols;
    hdr.step  = new_cols * new_cn * elemSize1();
    return hdr;
}

void cv::hip::HipMat::locateROI(Size& wholeSize, Point& ofs) const
{
    CV_DbgAssert(step > 0);
    size_t esz = elemSize();
    ptrdiff_t rawDiff = data - datastart;
    ofs.y  = (int)(rawDiff / step);
    ofs.x  = (int)((rawDiff - (ptrdiff_t)ofs.y * step) / esz);
    wholeSize.height = (int)((dataend - datastart + step - esz) / step);
    wholeSize.width  = (int)((step + esz - 1) / esz);
}

HipMat& cv::hip::HipMat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    Size wholeSize; Point ofs;
    locateROI(wholeSize, ofs);
    size_t esz = elemSize();
    int row1 = std::max(ofs.y - dtop,  0);
    int row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
    int col1 = std::max(ofs.x - dleft, 0);
    int col2 = std::min(ofs.x + cols + dright,  wholeSize.width);
    data    += (row1 - ofs.y) * (ptrdiff_t)step + (col1 - ofs.x) * (ptrdiff_t)esz;
    rows     = row2 - row1;
    cols     = col2 - col1;
    dataend  = data + step * (rows - 1) + esz * cols;
    updateContinuityFlag();
    return *this;
}

HipMat cv::hip::HipMat::row(int y)              const { return HipMat(*this, Range(y, y+1), Range::all()); }
HipMat cv::hip::HipMat::col(int x)              const { return HipMat(*this, Range::all(), Range(x, x+1)); }
HipMat cv::hip::HipMat::rowRange(int s, int e)  const { return HipMat(*this, Range(s, e), Range::all()); }
HipMat cv::hip::HipMat::rowRange(Range r)        const { return HipMat(*this, r, Range::all()); }
HipMat cv::hip::HipMat::colRange(int s, int e)  const { return HipMat(*this, Range::all(), Range(s, e)); }
HipMat cv::hip::HipMat::colRange(Range r)        const { return HipMat(*this, Range::all(), r); }
HipMat cv::hip::HipMat::operator()(Range rr, Range cr) const { return HipMat(*this, rr, cr); }
HipMat cv::hip::HipMat::operator()(Rect roi)     const { return HipMat(*this, roi); }
HipMat cv::hip::HipMat::clone()                  const { HipMat m; copyTo(m); return m; }

// ======================== Property accessors ========================

bool   cv::hip::HipMat::isContinuous() const { return (flags & Mat::CONTINUOUS_FLAG) != 0; }
size_t cv::hip::HipMat::elemSize()     const { return CV_ELEM_SIZE(flags); }
size_t cv::hip::HipMat::elemSize1()    const { return CV_ELEM_SIZE1(flags); }
int    cv::hip::HipMat::type()         const { return CV_MAT_TYPE(flags); }
int    cv::hip::HipMat::depth()        const { return CV_MAT_DEPTH(flags); }
int    cv::hip::HipMat::channels()     const { return CV_MAT_CN(flags); }
size_t cv::hip::HipMat::step1()        const { return step / elemSize1(); }
Size   cv::hip::HipMat::size()         const { return Size(cols, rows); }
bool   cv::hip::HipMat::empty()        const { return data == nullptr; }
void*  cv::hip::HipMat::cudaPtr()      const { return data; }

uchar*       cv::hip::HipMat::ptr(int y)       { return data + y * step; }
const uchar* cv::hip::HipMat::ptr(int y) const { return data + y * step; }

// ======================== HipMatND ========================

cv::hip::HipMatND::HipMatND(const MatShape& _shape, int _type)
    : flags(0), dims(0), data(nullptr), offset(0)
{
    create(_shape, _type);
}

cv::hip::HipMatND::~HipMatND() = default;

void cv::hip::HipMatND::setFields(MatShape _size, int _type, StepArray _step)
{
    _type &= Mat::TYPE_MASK;
    const size_t esz = (size_t)CV_ELEM_SIZE(_type);
    flags = Mat::MAGIC_VAL + _type;
    dims = static_cast<int>(_size.size());
    size = std::move(_size);
    if (_step.empty()) {
        step = StepArray(dims);
        step.back() = esz;
        for (int i = dims - 2; i >= 0; --i)
            step[(size_t)i] = step[(size_t)i + 1] * (size_t)size[(size_t)i + 1];
        flags |= Mat::CONTINUOUS_FLAG;
    } else {
        step = std::move(_step);
        if (step.size() < size.size())
            step.push_back(esz);
        bool continuous = true;
        for (int i = dims - 2; i >= 0; --i) {
            if (step[(size_t)i] != step[(size_t)i + 1] * (size_t)size[(size_t)i + 1]) {
                continuous = false;
                break;
            }
        }
        if (continuous)
            flags |= Mat::CONTINUOUS_FLAG;
        else
            flags &= ~Mat::CONTINUOUS_FLAG;
    }
    CV_Assert(size.size() == step.size());
    CV_Assert(step.back() == esz);
}

// ======================== HipData ========================

cv::hip::HipData::HipData(size_t size_) : data(nullptr), size(size_)
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipSafeCall(hipMalloc((void**)&data, size));
#endif
}

cv::hip::HipData::~HipData()
{
#ifdef HAVE_HIP
    if (data) hipSafeCall(hipFree(data));
#endif
}

// ======================== createContinuous / ensureSizeIsEnough ========================

void cv::hip::createContinuous(int rows, int cols, int type, OutputArray arr)
{
#ifndef HAVE_HIP
    CV_UNUSED(rows); CV_UNUSED(cols); CV_UNUSED(type); CV_UNUSED(arr); throw_no_hip();
#else
    bool needCreate = arr.empty() || arr.type() != type ||
                      !arr.isContinuous() || arr.rows() != 1 ||
                      arr.cols() != rows * cols;
    if (needCreate)
        arr.create(1, rows * cols, type);
#endif
}

void cv::hip::ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr)
{
#ifndef HAVE_HIP
    CV_UNUSED(rows); CV_UNUSED(cols); CV_UNUSED(type); CV_UNUSED(arr); throw_no_hip();
#else
    arr.create(rows, cols, type);
#endif
}

// ======================== BufferPool ========================

cv::hip::BufferPool::BufferPool(Stream& stream_)
{
#ifndef HAVE_HIP
    CV_UNUSED(stream_); throw_no_hip();
#else
    (void)stream_;
    allocator_ = Ptr<HipMat::Allocator>(HipMat::getStdAllocator(), [](HipMat::Allocator*){});
#endif
}

HipMat cv::hip::BufferPool::getBuffer(int rows, int cols, int type)
{
#ifndef HAVE_HIP
    CV_UNUSED(rows); CV_UNUSED(cols); CV_UNUSED(type); throw_no_hip();
#else
    HipMat m;
    m.allocator = allocator_.get();
    m.create(rows, cols, type);
    return m;
#endif
}

void cv::hip::setBufferPoolUsage(bool on)        { CV_UNUSED(on); }
void cv::hip::setBufferPoolConfig(int d, size_t s, int c) { CV_UNUSED(d); CV_UNUSED(s); CV_UNUSED(c); }

// ======================== HostMem ========================

#ifdef HAVE_HIP
namespace {

class HostMemAllocator CV_FINAL : public MatAllocator
{
public:
    explicit HostMemAllocator(unsigned int hipFlags_) : hipFlags(hipFlags_) {}

    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step,
                       AccessFlag, UMatUsageFlags) const CV_OVERRIDE
    {
        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--) {
            if (step) step[i] = total;
            total *= sizes[i];
        }
        void* ptr = nullptr;
        hipSafeCall(hipHostMalloc(&ptr, total, hipFlags));
        if (data) std::memcpy(ptr, data, total);
        UMatData* u   = new UMatData(this);
        u->data = u->origdata = (uchar*)ptr;
        u->size = total;
        u->flags = static_cast<UMatData::MemoryFlag>(0);
        return u;
    }

    bool allocate(UMatData*, AccessFlag, UMatUsageFlags) const CV_OVERRIDE { return false; }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        CV_Assert(u->urefcount == 0 && u->refcount == 0);
        hipSafeCall(hipHostFree(u->origdata));
        delete u;
    }

private:
    unsigned int hipFlags;
};

HostMemAllocator pageLocked(hipHostMallocDefault);
HostMemAllocator shared(hipHostMallocMapped);
HostMemAllocator writeCombined(hipHostMallocWriteCombined);

} // anonymous
#endif

MatAllocator* cv::hip::HostMem::getAllocator(HostMem::AllocType alloc_type)
{
#ifndef HAVE_HIP
    CV_UNUSED(alloc_type); throw_no_hip();
#else
    switch (alloc_type) {
        case PAGE_LOCKED:    return &pageLocked;
        case SHARED:         return &shared;
        case WRITE_COMBINED: return &writeCombined;
    }
    return &pageLocked;
#endif
}

cv::hip::HostMem::HostMem(HostMem::AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      alloc_type(alloc_type_)
{}

cv::hip::HostMem::HostMem(const HostMem& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step),
      data(m.data), refcount(m.refcount),
      datastart(m.datastart), dataend(m.dataend),
      alloc_type(m.alloc_type)
{
    if (refcount) CV_XADD(refcount, 1);
}

cv::hip::HostMem::HostMem(int rows_, int cols_, int type_, HostMem::AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      alloc_type(alloc_type_)
{
    if (rows_ > 0 && cols_ > 0) create(rows_, cols_, type_);
}

cv::hip::HostMem::HostMem(Size size_, int type_, HostMem::AllocType alloc_type_)
    : HostMem(size_.height, size_.width, type_, alloc_type_)
{}

cv::hip::HostMem::HostMem(InputArray arr, HostMem::AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0),
      data(nullptr), refcount(nullptr),
      datastart(nullptr), dataend(nullptr),
      alloc_type(alloc_type_)
{
    Mat m = arr.getMat();
    create(m.rows, m.cols, m.type());
    m.copyTo(createMatHeader());
}

cv::hip::HostMem::~HostMem() { release(); }

HostMem& cv::hip::HostMem::operator=(const HostMem& m)
{
    if (this != &m) { HostMem tmp(m); swap(tmp); }
    return *this;
}

void cv::hip::HostMem::swap(HostMem& b)
{
    std::swap(flags,     b.flags);
    std::swap(rows,      b.rows);
    std::swap(cols,      b.cols);
    std::swap(step,      b.step);
    std::swap(data,      b.data);
    std::swap(refcount,  b.refcount);
    std::swap(datastart, b.datastart);
    std::swap(dataend,   b.dataend);
    std::swap(alloc_type, b.alloc_type);
}

HostMem cv::hip::HostMem::clone() const
{
    HostMem m(rows, cols, type(), alloc_type);
    createMatHeader().copyTo(m.createMatHeader());
    return m;
}

void cv::hip::HostMem::create(int rows_, int cols_, int type_)
{
#ifndef HAVE_HIP
    CV_UNUSED(rows_); CV_UNUSED(cols_); CV_UNUSED(type_); throw_no_hip();
#else
    type_ = CV_MAT_TYPE(type_);
    if (rows == rows_ && cols == cols_ && type() == type_ && data) return;
    release();
    if (rows_ == 0 || cols_ == 0) return;
    MatAllocator* a = getAllocator(alloc_type);
    int sz[2] = { rows_, cols_ };
    UMatData* u = a->allocate(2, sz, type_, nullptr, &step, ACCESS_READ | ACCESS_WRITE, USAGE_DEFAULT);
    data = datastart = u->data;
    rows  = rows_;
    cols  = cols_;
    flags = Mat::MAGIC_VAL | CV_MAT_CONT_FLAG | type_;
    refcount  = (int*)fastMalloc(sizeof(int));
    *refcount = 1;
    dataend = data + step * rows;
    delete u;
#endif
}

void cv::hip::HostMem::create(Size size_, int type_) { create(size_.height, size_.width, type_); }

HostMem cv::hip::HostMem::reshape(int cn, int rows_) const
{
    HostMem hdr = *this;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((cn - 1) << CV_CN_SHIFT);
    if (rows_ != 0) hdr.rows = rows_;
    return hdr;
}

void cv::hip::HostMem::release()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (refcount && CV_XADD(refcount, -1) == 1) {
        hipSafeCall(hipHostFree(datastart));
        fastFree(refcount);
    }
    data = datastart = nullptr;
    dataend  = nullptr;
    rows = cols = 0;
    step = 0; flags = 0; refcount = nullptr;
#endif
}

Mat  cv::hip::HostMem::createMatHeader() const { return Mat(rows, cols, type(), data, step); }

HipMat cv::hip::HostMem::createHipMatHeader() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    HipMat m;
    hipSafeCall(hipHostGetDevicePointer((void**)&m.data, data, 0));
    m.flags    = flags;
    m.rows     = rows;
    m.cols     = cols;
    m.step     = step;
    m.datastart = m.data;
    m.dataend   = m.data + step * rows;
    return m;
#endif
}

bool   cv::hip::HostMem::isContinuous() const { return (flags & Mat::CONTINUOUS_FLAG) != 0; }
size_t cv::hip::HostMem::elemSize()     const { return CV_ELEM_SIZE(flags); }
size_t cv::hip::HostMem::elemSize1()    const { return CV_ELEM_SIZE1(flags); }
int    cv::hip::HostMem::type()         const { return CV_MAT_TYPE(flags); }
int    cv::hip::HostMem::depth()        const { return CV_MAT_DEPTH(flags); }
int    cv::hip::HostMem::channels()     const { return CV_MAT_CN(flags); }
size_t cv::hip::HostMem::step1()        const { return step / elemSize1(); }
Size   cv::hip::HostMem::size()         const { return Size(cols, rows); }
bool   cv::hip::HostMem::empty()        const { return data == nullptr; }

// ======================== Stream::Impl ========================

#ifdef HAVE_HIP
struct cv::hip::Stream::Impl
{
    hipStream_t stream;
    bool ownStream;
    Ptr<HipMat::Allocator> allocator;

    Impl() : stream(0), ownStream(false) {}

    explicit Impl(const Ptr<HipMat::Allocator>& alloc)
        : stream(0), ownStream(true), allocator(alloc)
    {
        hipSafeCall(hipStreamCreate(&stream));
    }

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

cv::hip::Stream::Stream(const Ptr<HipMat::Allocator>& allocator_)
{
#ifndef HAVE_HIP
    CV_UNUSED(allocator_); throw_no_hip();
#else
    impl = makePtr<Impl>(allocator_);
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
