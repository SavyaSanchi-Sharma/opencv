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
        class CV_EXPORTS_W HipMat{
            public:
                class CV_EXPORTS_W Allocator{
                    public: 
                        virtual ~Allocator() {}
                        virtual bool allocate(HipMat* mat, int rows, int cols, size_t elemSize) = 0;
                        virtual void free(HipMat* mat) = 0;
                };
        CV_WRAP static HipMat::Allocator* defaultAllocator();
        CV_WRAP static void setDefaultAllocator(HipMat::Allocator* allocator);
        CV_WRAP static HipMat::Allocator* getStdAllocator();

        CV_WRAP explicit HipMat(HipMat::Allocator* allocator = HipMat::defaultAllocator());
        CV_WRAP HipMat(int rows, int cols, int type, HipMat::Allocator* allocator = HipMat::defaultAllocator());
        CV_WRAP HipMat(Size size, int type, HipMat::Allocator* allocator = HipMat::defaultAllocator());
        CV_WRAP HipMat(int rows, int cols, int type, Scalar s, HipMat::Allocator* allocator = HipMat::defaultAllocator());
        CV_WRAP HipMat(Size size, int type, Scalar s, HipMat::Allocator* allocator = HipMat::defaultAllocator());
        CV_WRAP HipMat(const HipMat& m);
        HipMat(int rows, int cols, int type, void* data, size_t step = Mat::AUTO_STEP);
        HipMat(Size size, int type, void* data, size_t step = Mat::AUTO_STEP);
        CV_WRAP HipMat(const HipMat& m, Range rowRange, Range colRange);
        CV_WRAP HipMat(const HipMat& m, Rect roi);
        CV_WRAP explicit HipMat(InputArray arr, HipMat::Allocator* allocator = HipMat::defaultAllocator());
        ~HipMat();
        HipMat& operator =(const HipMat& m);
        CV_WRAP void create(int rows, int cols, int type);
        CV_WRAP void create(Size size, int type);
        CV_WRAP void release();
        void fit(int rows,int cols,int type);
        void fit(Size size,int type);
        CV_WRAP void swap(HipMat& mat);
        CV_WRAP void upload(InputArray arr);
        CV_WRAP void upload(InputArray arr, Stream& stream);
        CV_WRAP void download(OutputArray arr) const;
        CV_WRAP void download(OutputArray arr, Stream& stream) const;
        CV_WRAP HipMat clone() const;
        void copyTo(OutputArray dst) const;
        CV_WRAP void copyTo(CV_OUT HipMat& dst) const;
        CV_WRAP void copyTo(OutputArray dst, Stream& stream) const;
        CV_WRAP void copyTo(CV_OUT HipMat& dst, Stream& stream) const;
        void copyTo(OutputArray dst, InputArray mask) const;
        CV_WRAP void copyTo(CV_OUT HipMat& dst, InputArray mask) const;

        void copyTo(OutputArray dst, InputArray mask, Stream& stream) const;
        CV_WRAP void copyTo(CV_OUT HipMat& dst, HipMat& mask, Stream& stream) const;
        CV_WRAP HipMat& setTo(Scalar s);
        CV_WRAP HipMat& setTo(Scalar s, Stream& stream);
        CV_WRAP HipMat& setTo(Scalar s, InputArray mask);
        CV_WRAP HipMat& setTo(Scalar s, InputArray mask, Stream& stream);

        void convertTo(OutputArray dst, int rtype) const;
        CV_WRAP void convertTo(CV_OUT HipMat& dst, int rtype) const;

        void convertTo(OutputArray dst, int rtype, Stream& stream) const;
        CV_WRAP void convertTo(CV_OUT HipMat& dst, int rtype, Stream& stream) const;

        void convertTo(OutputArray dst, int rtype, double alpha, double beta = 0.0) const;

    #ifdef OPENCV_BINDINGS_PARSER
        CV_WRAP void convertTo(CV_OUT HipMat& dst, int rtype, double alpha=1.0, double beta = 0.0) const;
    #endif

        void convertTo(OutputArray dst, int rtype, double alpha, Stream& stream) const;

        void convertTo(OutputArray dst, int rtype, double alpha, double beta, Stream& stream) const;
        CV_WRAP void convertTo(CV_OUT HipMat& dst, int rtype, double alpha, double beta, Stream& stream) const;

        CV_WRAP void assignTo(HipMat& m, int type = -1) const;

        //! returns pointer to y-th row
        uchar* ptr(int y = 0);
        const uchar* ptr(int y = 0) const;

        //! template version of the above method
        template<typename _Tp> _Tp* ptr(int y = 0);
        template<typename _Tp> const _Tp* ptr(int y = 0) const;

        template <typename _Tp> operator PtrStepSz<_Tp>() const;
        template <typename _Tp> operator PtrStep<_Tp>() const;

        //! returns a new HipMat header for the specified row
        CV_WRAP HipMat row(int y) const;

        //! returns a new HipMat header for the specified column
        CV_WRAP HipMat col(int x) const;

        //! ... for the specified row span
        CV_WRAP HipMat rowRange(int startrow, int endrow) const;
        CV_WRAP HipMat rowRange(Range r) const;

        //! ... for the specified column span
        CV_WRAP HipMat colRange(int startcol, int endcol) const;
        CV_WRAP HipMat colRange(Range r) const;

        //! extracts a rectangular sub-HipMat (this is a generalized form of row, rowRange etc.)
        HipMat operator ()(Range rowRange, Range colRange) const;
        HipMat operator ()(Rect roi) const;

        //! creates alternative HipMat header for the same data, with different
        //! number of channels and/or different number of rows
        CV_WRAP HipMat reshape(int cn, int rows = 0) const;

        //! locates HipMat header within a parent HipMat
        CV_WRAP void locateROI(Size& wholeSize, Point& ofs) const;

        //! moves/resizes the current HipMat ROI inside the parent HipMat
        CV_WRAP HipMat& adjustROI(int dtop, int dbottom, int dleft, int dright);

        //! returns true iff the HipMat data is continuous
        //! (i.e. when there are no gaps between successive rows)
        CV_WRAP bool isContinuous() const;

        //! returns element size in bytes
        CV_WRAP size_t elemSize() const;

        //! returns the size of element channel in bytes
        CV_WRAP size_t elemSize1() const;

        //! returns element type
        CV_WRAP int type() const;

        //! returns element type
        CV_WRAP int depth() const;

        //! returns number of channels
        CV_WRAP int channels() const;

        //! returns step/elemSize1()
        CV_WRAP size_t step1() const;

        //! returns HipMat size : width == number of columns, height == number of rows
        CV_WRAP Size size() const;

        //! returns true if HipMat data is NULL
        CV_WRAP bool empty() const;

        // returns pointer to cuda memory
        CV_WRAP void* cudaPtr() const;

        //! internal use method: updates the continuity flag
        CV_WRAP void updateContinuityFlag();

        /*! includes several bit-fields:
        - the magic signature
        - continuity flag
        - depth
        - number of channels
        */
        int flags;

        //! the number of rows and columns
        int rows, cols;

        //! a distance between successive rows in bytes; includes the gap if any
        CV_PROP size_t step;

        //! pointer to the data
        uchar* data;

        //! pointer to the reference counter;
        //! when HipMat points to user-allocated data, the pointer is NULL
        int* refcount;

        //! helper fields used in locateROI and adjustROI
        uchar* datastart;
        const uchar* dataend;

        //! allocator
        Allocator* allocator;

        //! UMat backing — handle is the device pointer (set by HipAllocator)
        UMatData* u;
    };

    struct CV_EXPORTS_W HipData
    {
        explicit HipData(size_t _size);
        ~HipData();

        HipData(const HipData&) = delete;
        HipData& operator=(const HipData&) = delete;

        HipData(HipData&&) = delete;
        HipData& operator=(HipData&&) = delete;

        uchar* data;
        size_t size;
    };

    class CV_EXPORTS_W HipMatND
    {
    public:
        using SizeArray = MatShape;
        using StepArray = std::vector<size_t>;
        using IndexArray = std::vector<int>;

        //! destructor
        ~HipMatND();

        //! default constructor
        HipMatND();

        /** @overload
        @param shape Array of integers specifying an n-dimensional array shape.
        @param type Array type. Use CV_8UC1, ..., CV_16FC4 to create 1-4 channel matrices, or
        CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
        */
        HipMatND(const MatShape& shape, int type);

        /** @overload
        @param shape Array of integers specifying an n-dimensional array shape.
        @param type Array type. Use CV_8UC1, ..., CV_16FC4 to create 1-4 channel matrices, or
        CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
        @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
        allocate matrix data. Instead, they just initialize the matrix header that points to the specified
        data, which means that no data is copied. This operation is very efficient and can be used to
        process external data using OpenCV functions. The external data is not automatically deallocated, so
        you should take care of it.
        @param step Array of shape.size() or shape.size()-1 steps in case of a multi-dimensional array
        (if specified, the last step must be equal to the element size, otherwise it will be added as such).
        If not specified, the matrix is assumed to be continuous.
        */
        HipMatND(const MatShape& shape, int type, void* data, StepArray step = StepArray());

        /** @brief Allocates GPU memory.
        Suppose there is some GPU memory already allocated. In that case, this method may choose to reuse that
        GPU memory under the specific condition: it must be of the same size and type, not externally allocated,
        the GPU memory is continuous(i.e., isContinuous() is true), and is not a sub-matrix of another HipMatND
        (i.e., isSubmatrix() is false). In other words, this method guarantees that the GPU memory allocated by
        this method is always continuous and is not a sub-region of another HipMatND.
        */
        void create(const MatShape& shape, int type);

        void release();

        void swap(HipMatND& m) noexcept;

        /** @brief Allocates or reuses underlying storage to fit the requested n-D size and type.
        - No-op if already compatible (sufficient capacity, continuous, not a submatrix, not external, no ROI).
        - Reallocates otherwise.
        Mirrors 2D HipMat::fit semantics for multi-dimensional tensors.
        */
        void fit(const MatShape& shape, int type);

        /** @brief Creates a full copy of the array and the underlying data.
        The method creates a full copy of the array. It mimics the behavior of Mat::clone(), i.e.
        the original step is not taken into account. So, the array copy is a continuous array
        occupying total()\*elemSize() bytes.
        */
        HipMatND clone() const;

        /** @overload
        This overload is non-blocking, so it may return even if the copy operation is not finished.
        */
        HipMatND clone(Stream& stream) const;

        /** @brief Extracts a sub-matrix.
        The operator makes a new header for the specified sub-array of \*this.
        The operator is an O(1) operation, that is, no matrix data is copied.
        @param ranges Array of selected ranges along each dimension.
        */
        HipMatND operator()(const std::vector<Range>& ranges) const;

        /** @brief Creates a HipMat header for a 2D plane part of an n-dim matrix.
        @note The returned HipMat is constructed with the constructor for user-allocated data.
        That is, It does not perform reference counting.
        @note This function does not increment this HipMatND's reference counter.
        */
        HipMat createHipMatHeader(IndexArray idx, Range rowRange, Range colRange) const;

        /** @overload
        Creates a HipMat header if this HipMatND is effectively 2D.
        @note The returned HipMat is constructed with the constructor for user-allocated data.
        That is, It does not perform reference counting.
        @note This function does not increment this HipMatND's reference counter.
        */
        HipMat createHipMatHeader() const;

        /** @brief Extracts a 2D plane part of an n-dim matrix.
        It differs from createHipMatHeader(IndexArray, Range, Range) in that it clones a part of this
        HipMatND to the returned HipMat.
        @note This operator does not increment this HipMatND's reference counter;
        */
        HipMat operator()(IndexArray idx, Range rowRange, Range colRange) const;

        /** @brief Extracts a 2D plane part of an n-dim matrix if this HipMatND is effectively 2D.
        It differs from createHipMatHeader() in that it clones a part of this HipMatND.
        @note This operator does not increment this HipMatND's reference counter;
        */
        operator HipMat() const;

        HipMatND(const HipMatND&) = default;
        HipMatND& operator=(const HipMatND&) = default;

        HipMatND(HipMatND&&) = default;
        HipMatND& operator=(HipMatND&&) = default;

        void upload(InputArray src);
        void upload(InputArray src, Stream& stream);
        void download(OutputArray dst) const;
        void download(OutputArray dst, Stream& stream) const;

        //! returns true iff the HipMatND data is continuous
        //! (i.e. when there are no gaps between successive rows)
        bool isContinuous() const;

        //! returns true if the matrix is a sub-matrix of another matrix
        bool isSubmatrix() const;

        //! returns element size in bytes
        size_t elemSize() const;

        //! returns the size of element channel in bytes
        size_t elemSize1() const;

        //! returns true if data is null
        bool empty() const;

        //! returns true if not empty and points to external(user-allocated) gpu memory
        bool external() const;

        //! returns pointer to the first byte of the GPU memory
        uchar* getDevicePtr() const;

        //! returns the total number of array elements
        size_t total() const;

        //! returns the size of underlying memory in bytes
        size_t totalMemSize() const;

        //! returns element type
        int type() const;

    private:
        //! internal use
        void setFields(MatShape size, int type, StepArray step = StepArray());

    public:
        /*! includes several bit-fields:
        - the magic signature
        - continuity flag
        - depth
        - number of channels
        */
        int flags;

        //! matrix dimensionality
        int dims;

        //! shape of this array
        MatShape size;

        /*! step values
        Their semantics is identical to the semantics of step for Mat.
        */
        StepArray step;

    private:
        /*! internal use
        If this HipMatND holds external memory, this is empty.
        */
        std::shared_ptr<HipData> data_;

        /*! internal use
        If this HipMatND manages memory with reference counting, this value is
        always equal to data_->data. If this HipMatND holds external memory,
        data_ is empty and data points to the external memory.
        */
        uchar* data;

        /*! internal use
        If this HipMatND is a sub-matrix of a larger matrix, this value is the
        difference of the first byte between the sub-matrix and the whole matrix.
        */
        size_t offset;
    };
    using SizeArray = HipMatND::SizeArray;
    using StepArray = HipMatND::StepArray;
    using IndexArray = HipMatND::IndexArray;

    CV_EXPORTS_W bool useHip();
    CV_EXPORTS MatAllocator* getHipAllocator();

    namespace device {
        CV_EXPORTS void setToWithoutMask(HipMat& mat, Scalar val, Stream& stream);
        CV_EXPORTS void setToWithMask(HipMat& mat, const HipMat& mask, Scalar val, Stream& stream);
        CV_EXPORTS void copyToWithMask(const HipMat& src, HipMat& dst, const HipMat& mask, Stream& stream);
        CV_EXPORTS void convertToNoScale(const HipMat& src, HipMat& dst, Stream& stream);
        CV_EXPORTS void convertToScale(const HipMat& src, HipMat& dst, double alpha, double beta, Stream& stream);
        CV_EXPORTS void multiplyF32(const HipMat& src1, const HipMat& src2, HipMat& dst, Stream& stream);
    } // namespace device

    CV_EXPORTS_W void createContinuous(int rows, int cols, int type, OutputArray arr);
    CV_EXPORTS_W void ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr);
    CV_EXPORTS_W HipMat inline createHipMatFromHipMemory(int rows, int cols, int type, size_t hipMemoryAddress, size_t step = Mat::AUTO_STEP) {
        return HipMat(rows,cols,type, reinterpret_cast<void*>(hipMemoryAddress), step);
    }
    CV_EXPORTS_W HipMat inline createHipMatFromHipMemory(Size size, int type, size_t hipMemoryAddress, size_t step = Mat::AUTO_STEP) {
        return HipMat(size,type, reinterpret_cast<void*>(hipMemoryAddress), step);
    }
    class CV_EXPORTS_W BufferPool{
        public: 
            CV_WRAP explicit BufferPool(Stream& stream);
            CV_WRAP HipMat getBuffer(int rows, int cols, int type);
// Warning : unreachable code using Ninja
#if defined _MSC_VER && _MSC_VER >= 1920
#pragma warning(push)
#pragma warning(disable: 4702)
#endif
    CV_WRAP HipMat getBuffer(Size size, int type) { return getBuffer(size.height, size.width, type); }
#if defined _MSC_VER && _MSC_VER >= 1920
#pragma warning(pop)
#endif
        CV_WRAP Ptr<HipMat::Allocator> getAllocator() const {return allocator_;}
        private: 
            Ptr<HipMat::Allocator> allocator_;
    };

    //! BufferPool management (must be called before Stream creation)
    CV_EXPORTS_W void setBufferPoolUsage(bool on);
    CV_EXPORTS_W void setBufferPoolConfig(int deviceId, size_t stackSize, int stackCount);

    class CV_EXPORTS_W HostMem{
        public:
            enum AllocType{
                PAGE_LOCKED = 1,
                SHARED=2,
                WRITE_COMBINED = 4
            };
            static MatAllocator* getAllocator(HostMem::AllocType alloc_type=HostMem::AllocType::PAGE_LOCKED);
            CV_WRAP explicit HostMem(HostMem::AllocType alloc_type=HostMem::AllocType::PAGE_LOCKED);
            HostMem(const HostMem& m);

        CV_WRAP HostMem(int rows, int cols, int type, HostMem::AllocType alloc_type = HostMem::AllocType::PAGE_LOCKED);
        CV_WRAP HostMem(Size size, int type, HostMem::AllocType alloc_type = HostMem::AllocType::PAGE_LOCKED);

        CV_WRAP explicit HostMem(InputArray arr, HostMem::AllocType alloc_type = HostMem::AllocType::PAGE_LOCKED);

        ~HostMem();

        HostMem& operator =(const HostMem& m);

        //! swaps with other smart pointer
        CV_WRAP void swap(HostMem& b);

        //! returns deep copy of the matrix, i.e. the data is copied
        CV_WRAP HostMem clone() const;

        //! allocates new matrix data unless the matrix already has specified size and type.
        CV_WRAP void create(int rows, int cols, int type);
        void create(Size size, int type);

        //! creates alternative HostMem header for the same data, with different
        //! number of channels and/or different number of rows
        CV_WRAP HostMem reshape(int cn, int rows = 0) const;

        //! decrements reference counter and released memory if needed.
        void release();

        //! returns matrix header with disabled reference counting for HostMem data.
        CV_WRAP Mat createMatHeader() const;

        /** @brief Maps CPU memory to GPU address space and creates the cuda::HipMat header without reference counting
        for it.

        This can be done only if memory was allocated with the SHARED flag and if it is supported by the
        hardware. Laptops often share video and CPU memory, so address spaces can be mapped, which
        eliminates an extra copy.
        */
        HipMat createHipMatHeader() const;

        // Please see cv::Mat for descriptions
        CV_WRAP bool isContinuous() const;
        CV_WRAP size_t elemSize() const;
        CV_WRAP size_t elemSize1() const;
        CV_WRAP int type() const;
        CV_WRAP int depth() const;
        CV_WRAP int channels() const;
        CV_WRAP size_t step1() const;
        CV_WRAP Size size() const;
        CV_WRAP bool empty() const;

        // Please see cv::Mat for descriptions
        int flags;
        int rows, cols;
        CV_PROP size_t step;

        uchar* data;
        int* refcount;

        uchar* datastart;
        const uchar* dataend;

        AllocType alloc_type;

    };


    class CV_EXPORTS_W Stream{
        typedef void(Stream::*bool_type)() const;
        void this_type_does_not_support_comparisions() const {}
        public:
            typedef void(*StreamCallback)(int status,void* userData);
            CV_WRAP Stream();
            CV_WRAP Stream(const Ptr<HipMat::Allocator>& allocator);
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
                Stream(const Ptr<Impl>& impl);
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
        Event(const Ptr<Impl>& impl);

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

#ifndef OPENCV_CORE_HIP_IMPL
#include "opencv2/core/hip.inl.hpp"
#endif

#endif /*for OPENCV_CORE_HIP_HPP*/

