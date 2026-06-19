#ifndef OPENCV_CORE_HIPINL_HPP
#define OPENCV_CORE_HIPINL_HPP

#include "opencv2/core/hip.hpp"
namespace cv{
    namespace hip{
        inline 
        HipMat::HipMat(Allocator* allocator_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
        {}

        inline
        HipMat::HipMat(int rows_, int cols_, int type_, Allocator* allocator_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
        {
            if (rows_ > 0 && cols_ > 0)
                create(rows_, cols_, type_);
        }

        inline
        HipMat::HipMat(Size size_, int type_, Allocator* allocator_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
        {
            if (size_.height > 0 && size_.width > 0)
                create(size_.height, size_.width, type_);
        }

        // WARNING: unreachable code using Ninja
        #if defined _MSC_VER && _MSC_VER >= 1920
        #pragma warning(push)
        #pragma warning(disable: 4702)
        #endif
        inline
        HipMat::HipMat(int rows_, int cols_, int type_, Scalar s_, Allocator* allocator_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
        {
            if (rows_ > 0 && cols_ > 0)
            {
                create(rows_, cols_, type_);
                setTo(s_);
            }
        }

        inline
        HipMat::HipMat(Size size_, int type_, Scalar s_, Allocator* allocator_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
        {
            if (size_.height > 0 && size_.width > 0)
            {
                create(size_.height, size_.width, type_);
                setTo(s_);
            }
        }
        #if defined _MSC_VER && _MSC_VER >= 1920
        #pragma warning(pop)
        #endif

        inline
        HipMat::HipMat(const HipMat& m)
            : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), allocator(m.allocator)
        {
            if (refcount)
                CV_XADD(refcount, 1);
        }

        inline
        HipMat::HipMat(InputArray arr, Allocator* allocator_) :
            flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
        {
            upload(arr);
        }

        inline
        HipMat::~HipMat()
        {
            release();
        }

        inline
        HipMat& HipMat::operator =(const HipMat& m)
        {
            if (this != &m)
            {
                HipMat temp(m);
                swap(temp);
            }

            return *this;
        }

        inline
        void HipMat::create(Size size_, int type_)
        {
            create(size_.height, size_.width, type_);
        }

        inline
        void HipMat::swap(HipMat& b)
        {
            std::swap(flags, b.flags);
            std::swap(rows, b.rows);
            std::swap(cols, b.cols);
            std::swap(step, b.step);
            std::swap(data, b.data);
            std::swap(datastart, b.datastart);
            std::swap(dataend, b.dataend);
            std::swap(refcount, b.refcount);
            std::swap(allocator, b.allocator);
        }

        inline
        HipMat HipMat::clone() const
        {
            HipMat m;
            copyTo(m);
            return m;
        }

        // WARNING: unreachable code using Ninja
        #if defined _MSC_VER && _MSC_VER >= 1920
        #pragma warning(push)
        #pragma warning(disable: 4702)
        #endif
        inline
        void HipMat::copyTo(OutputArray dst, InputArray mask) const
        {
            copyTo(dst, mask, Stream::Null());
        }
        #if defined _MSC_VER && _MSC_VER >= 1920
        #pragma warning(pop)
        #endif

        inline
        HipMat& HipMat::setTo(Scalar s)
        {
            return setTo(s, Stream::Null());
        }

        inline
        HipMat& HipMat::setTo(Scalar s, InputArray mask)
        {
            return setTo(s, mask, Stream::Null());
        }

        // WARNING: unreachable code using Ninja
        #if defined _MSC_VER && _MSC_VER >= 1920
        #pragma warning(push)
        #pragma warning(disable: 4702)
        #endif
        inline
        void HipMat::convertTo(OutputArray dst, int rtype) const
        {
            convertTo(dst, rtype, Stream::Null());
        }

        inline
        void HipMat::convertTo(OutputArray dst, int rtype, double alpha, double beta) const
        {
            convertTo(dst, rtype, alpha, beta, Stream::Null());
        }
        #if defined _MSC_VER && _MSC_VER >= 1920
        #pragma warning(pop)
        #endif

        inline
        void HipMat::convertTo(OutputArray dst, int rtype, double alpha, Stream& stream) const
        {
            convertTo(dst, rtype, alpha, 0.0, stream);
        }

        inline
        void HipMat::assignTo(HipMat& m, int _type) const
        {
            if (_type < 0)
                m = *this;
            else
                convertTo(m, _type);
        }

        inline
        uchar* HipMat::ptr(int y)
        {
            CV_DbgAssert( (unsigned)y < (unsigned)rows );
            return data + step * y;
        }

        inline
        const uchar* HipMat::ptr(int y) const
        {
            CV_DbgAssert( (unsigned)y < (unsigned)rows );
            return data + step * y;
        }

        template<typename _Tp> inline
        _Tp* HipMat::ptr(int y)
        {
            return (_Tp*)ptr(y);
        }

        template<typename _Tp> inline
        const _Tp* HipMat::ptr(int y) const
        {
            return (const _Tp*)ptr(y);
        }

        template <class T> inline
        HipMat::operator PtrStepSz<T>() const
        {
            return PtrStepSz<T>(rows, cols, (T*)data, step);
        }

        template <class T> inline
        HipMat::operator PtrStep<T>() const
        {
            return PtrStep<T>((T*)data, step);
        }

        inline
        HipMat HipMat::row(int y) const
        {
            return HipMat(*this, Range(y, y+1), Range::all());
        }

        inline
        HipMat HipMat::col(int x) const
        {
            return HipMat(*this, Range::all(), Range(x, x+1));
        }

        inline
        HipMat HipMat::rowRange(int startrow, int endrow) const
        {
            return HipMat(*this, Range(startrow, endrow), Range::all());
        }

        inline
        HipMat HipMat::rowRange(Range r) const
        {
            return HipMat(*this, r, Range::all());
        }

        inline
        HipMat HipMat::colRange(int startcol, int endcol) const
        {
            return HipMat(*this, Range::all(), Range(startcol, endcol));
        }

        inline
        HipMat HipMat::colRange(Range r) const
        {
            return HipMat(*this, Range::all(), r);
        }

        inline
        HipMat HipMat::operator ()(Range rowRange_, Range colRange_) const
        {
            return HipMat(*this, rowRange_, colRange_);
        }

        inline
        HipMat HipMat::operator ()(Rect roi) const
        {
            return HipMat(*this, roi);
        }

        inline
        bool HipMat::isContinuous() const
        {
            return (flags & Mat::CONTINUOUS_FLAG) != 0;
        }

        inline
        size_t HipMat::elemSize() const
        {
            return CV_ELEM_SIZE(flags);
        }

        inline
        size_t HipMat::elemSize1() const
        {
            return CV_ELEM_SIZE1(flags);
        }

        inline
        int HipMat::type() const
        {
            return CV_MAT_TYPE(flags);
        }

        inline
        int HipMat::depth() const
        {
            return CV_MAT_DEPTH(flags);
        }

        inline
        int HipMat::channels() const
        {
            return CV_MAT_CN(flags);
        }

        inline
        size_t HipMat::step1() const
        {
            return step / elemSize1();
        }

        inline
        Size HipMat::size() const
        {
            return Size(cols, rows);
        }

        inline
        bool HipMat::empty() const
        {
            return data == 0;
        }

        inline
        void* HipMat::cudaPtr() const
        {
            return data;
        }

        static inline
        HipMat createContinuous(int rows, int cols, int type)
        {
            HipMat m;
            m.create(rows, cols, type);
            return m;
        }

        static inline
        void createContinuous(Size size, int type, OutputArray arr)
        {
            createContinuous(size.height, size.width, type, arr);
        }

        static inline
        HipMat createContinuous(Size size, int type)
        {
            return createContinuous(size.height, size.width, type);
        }

        static inline
        void ensureSizeIsEnough(Size size, int type, OutputArray arr)
        {
            ensureSizeIsEnough(size.height, size.width, type, arr);
        }

        static inline
        void swap(HipMat& a, HipMat& b)
        {
            a.swap(b);
        }

        //===================================================================================
        // HipMatND
        //===================================================================================

        inline
        HipMatND::HipMatND() :
            flags(0), dims(0), data(nullptr), offset(0)
        {
        }

        inline
        HipMatND::HipMatND(const MatShape& _shape, int _type) :
            flags(0), dims(0), data(nullptr), offset(0)
        {
            create(_shape, _type);
        }

        inline
        void HipMatND::swap(HipMatND& m) noexcept
        {
            std::swap(*this, m);
        }

        inline
        bool HipMatND::isContinuous() const
        {
            return (flags & Mat::CONTINUOUS_FLAG) != 0;
        }

        inline
        bool HipMatND::isSubmatrix() const
        {
            return (flags & Mat::SUBMATRIX_FLAG) != 0;
        }

        inline
        size_t HipMatND::elemSize() const
        {
            return CV_ELEM_SIZE(flags);
        }

        inline
        size_t HipMatND::elemSize1() const
        {
            return CV_ELEM_SIZE1(flags);
        }

        inline
        bool HipMatND::empty() const
        {
            return data == nullptr;
        }

        inline
        bool HipMatND::external() const
        {
            return !empty() && data_.use_count() == 0;
        }

        inline
        uchar* HipMatND::getDevicePtr() const
        {
            return data + offset;
        }

        inline
        size_t HipMatND::total() const
        {
            size_t p = 1;
            for(auto s : size)
                p *= s;
            return p;
        }

        inline
        size_t HipMatND::totalMemSize() const
        {
            return size[0] * step[0];
        }

        inline
        int HipMatND::type() const
        {
            return CV_MAT_TYPE(flags);
        }

        //===================================================================================
        // HostMem
        //===================================================================================

        inline
        HostMem::HostMem(AllocType alloc_type_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
        {
        }

        inline
        HostMem::HostMem(const HostMem& m)
            : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), alloc_type(m.alloc_type)
        {
            if( refcount )
                CV_XADD(refcount, 1);
        }

        inline
        HostMem::HostMem(int rows_, int cols_, int type_, AllocType alloc_type_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
        {
            if (rows_ > 0 && cols_ > 0)
                create(rows_, cols_, type_);
        }

        inline
        HostMem::HostMem(Size size_, int type_, AllocType alloc_type_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
        {
            if (size_.height > 0 && size_.width > 0)
                create(size_.height, size_.width, type_);
        }

        inline
        HostMem::HostMem(InputArray arr, AllocType alloc_type_)
            : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
        {
            Mat m = arr.getMat();
            create(m.rows, m.cols, m.type());
            m.copyTo(createMatHeader());
        }

        inline
        HostMem::~HostMem()
        {
            release();
        }

        inline
        HostMem& HostMem::operator =(const HostMem& m)
        {
            if (this != &m)
            {
                HostMem temp(m);
                swap(temp);
            }

            return *this;
        }

        inline
        void HostMem::swap(HostMem& b)
        {
            std::swap(flags, b.flags);
            std::swap(rows, b.rows);
            std::swap(cols, b.cols);
            std::swap(step, b.step);
            std::swap(data, b.data);
            std::swap(datastart, b.datastart);
            std::swap(dataend, b.dataend);
            std::swap(refcount, b.refcount);
            std::swap(alloc_type, b.alloc_type);
        }

        inline
        HostMem HostMem::clone() const
        {
            HostMem m(size(), type(), alloc_type);
            createMatHeader().copyTo(m.createMatHeader());
            return m;
        }

        inline
        void HostMem::create(Size size_, int type_)
        {
            create(size_.height, size_.width, type_);
        }

        inline
        Mat HostMem::createMatHeader() const
        {
            return Mat(size(), type(), data, step);
        }

        inline
        bool HostMem::isContinuous() const
        {
            return (flags & Mat::CONTINUOUS_FLAG) != 0;
        }

        inline
        size_t HostMem::elemSize() const
        {
            return CV_ELEM_SIZE(flags);
        }

        inline
        size_t HostMem::elemSize1() const
        {
            return CV_ELEM_SIZE1(flags);
        }

        inline
        int HostMem::type() const
        {
            return CV_MAT_TYPE(flags);
        }

        inline
        int HostMem::depth() const
        {
            return CV_MAT_DEPTH(flags);
        }

        inline
        int HostMem::channels() const
        {
            return CV_MAT_CN(flags);
        }

        inline
        size_t HostMem::step1() const
        {
            return step / elemSize1();
        }

        inline
        Size HostMem::size() const
        {
            return Size(cols, rows);
        }

        inline
        bool HostMem::empty() const
        {
            return data == 0;
        }

        static inline
        void swap(HostMem& a, HostMem& b)
        {
            a.swap(b);
        }

        //===================================================================================
        // Stream
        //===================================================================================

        inline
        Stream::Stream(const Ptr<Impl>& impl_)
            : impl(impl_)
        {
        }

        //===================================================================================
        // Event
        //===================================================================================

        inline
        Event::Event(const Ptr<Impl>& impl)
            : impl_(impl)
        {
        }
    }
}





#endif /*for OPENCV_CORE_HIPINL_HPP*/


