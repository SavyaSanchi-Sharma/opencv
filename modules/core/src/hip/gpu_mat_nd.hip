// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #include "opencv2/opencv_modules.hpp"

// #ifndef HAVE_OPENCV_HIP

// #error "opencv_cudev is required"

// #else

#include "opencv2/core/hip.hpp"
#include "opencv2/core/hip_stream_accessor.hpp"
#include "opencv2/core/hipdev.hpp"

using namespace cv;
using namespace cv::hip;

/////////////////////////////////////////////////////
/// create

void HipMatND::create(const MatShape& _shape, int _type)
{
    {
        auto elements_nonzero = [](const MatShape& v)
        {
            return std::all_of(v.begin(), v.end(), [](int u){ return u > 0; });
        };
        CV_Assert(!_shape.empty());
        CV_Assert(elements_nonzero(_shape));
    }

    _type &= Mat::TYPE_MASK;

    if (size == _shape && type() == _type && !empty() && !external() && isContinuous() && !isSubmatrix())
        return;

    release();

    setFields(_shape, _type);

    data_ = std::make_shared<HipData>(totalMemSize());
    data = data_->data;
    offset = 0;
}

/////////////////////////////////////////////////////
/// release

void HipMatND::release()
{
    data = nullptr;
    data_.reset();

    flags = dims = offset = 0;
    size.clear();
    step.clear();
}

/////////////////////////////////////////////////////
/// clone

static bool next(uchar*& d, const uchar*& s, std::vector<int>& idx, const int dims, const HipMatND& dst, const HipMatND& src)
{
    int inc = dims-3;

    while (true)
    {
        if (idx[inc] == src.size[inc] - 1)
        {
            if (inc == 0)
            {
                return false;
            }

            idx[inc] = 0;
            d -= (dst.size[inc] - 1) * dst.step[inc];
            s -= (src.size[inc] - 1) * src.step[inc];
            inc--;
        }
        else
        {
            idx[inc]++;
            d += dst.step[inc];
            s += src.step[inc];
            break;
        }
    }

    return true;
}

HipMatND HipMatND::clone() const
{
    CV_DbgAssert(!empty());

    HipMatND ret(size, type());

    if (isContinuous())
    {
        CV_HIP_SAFE_CALL(hipMemcpy(ret.getDevicePtr(), getDevicePtr(), ret.totalMemSize(), hipMemcpyDeviceToDevice));
    }
    else
    {
        // 1D arrays are always continuous

        if (dims == 2)
        {
            CV_HIP_SAFE_CALL(
                hipMemcpy2D(ret.getDevicePtr(), ret.step[0], getDevicePtr(), step[0],
                    size[1]*step[1], size[0], hipMemcpyDeviceToDevice)
            );
        }
        else
        {
            std::vector<int> idx(dims-2, 0);

            uchar* d = ret.getDevicePtr();
            const uchar* s = getDevicePtr();

            // iterate each 2D plane
            do
            {
                CV_HIP_SAFE_CALL(
                    hipMemcpy2DAsync(
                        d, ret.step[dims-2], s, step[dims-2],
                        size[dims-1]*step[dims-1], size[dims-2], hipMemcpyDeviceToDevice)
                );
            }
            while (next(d, s, idx, dims, ret, *this));

            CV_HIP_SAFE_CALL(hipStreamSynchronize(0));
        }
    }

    return ret;
}

HipMatND HipMatND::clone(Stream& stream) const
{
    CV_DbgAssert(!empty());

    HipMatND ret(size, type());

    hipStream_t _stream = StreamAccessor::getStream(stream);

    if (isContinuous())
    {
        CV_HIP_SAFE_CALL(hipMemcpyAsync(ret.getDevicePtr(), getDevicePtr(), ret.totalMemSize(), hipMemcpyDeviceToDevice, _stream));
    }
    else
    {
        // 1D arrays are always continuous

        if (dims == 2)
        {
            CV_HIP_SAFE_CALL(
                hipMemcpy2DAsync(ret.getDevicePtr(), ret.step[0], getDevicePtr(), step[0],
                    size[1]*step[1], size[0], hipMemcpyDeviceToDevice, _stream)
            );
        }
        else
        {
            std::vector<int> idx(dims-2, 0);

            uchar* d = ret.getDevicePtr();
            const uchar* s = getDevicePtr();

            // iterate each 2D plane
            do
            {
                CV_HIP_SAFE_CALL(
                    hipMemcpy2DAsync(
                        d, ret.step[dims-2], s, step[dims-2],
                        size[dims-1]*step[dims-1], size[dims-2], hipMemcpyDeviceToDevice, _stream)
                );
            }
            while (next(d, s, idx, dims, ret, *this));
        }
    }

    return ret;
}

/////////////////////////////////////////////////////
/// upload

void HipMatND::upload(InputArray src)
{
    Mat mat = src.getMat();

    CV_DbgAssert(!mat.empty());

    if (!mat.isContinuous())
        mat = mat.clone();

    MatShape _shape(mat.size.p, mat.size.p + mat.dims);

    create(_shape, mat.type());

    CV_HIP_SAFE_CALL(hipMemcpy(getDevicePtr(), mat.data, totalMemSize(), hipMemcpyHostToDevice));
}

void HipMatND::upload(InputArray src, Stream& stream)
{
    Mat mat = src.getMat();

    CV_DbgAssert(!mat.empty());

    if (!mat.isContinuous())
        mat = mat.clone();

    MatShape _shape(mat.size.p, mat.size.p + mat.dims);

    create(_shape, mat.type());

    hipStream_t _stream = StreamAccessor::getStream(stream);
    CV_HIP_SAFE_CALL(hipMemcpyAsync(getDevicePtr(), mat.data, totalMemSize(), hipMemcpyHostToDevice, _stream));
}

/////////////////////////////////////////////////////
/// download

void HipMatND::download(OutputArray dst) const
{
    CV_DbgAssert(!empty());

    dst.create(dims, size.data(), type());
    Mat mat = dst.getMat();

    HipMatND gmat = *this;

    if (!gmat.isContinuous())
        gmat = gmat.clone();

    CV_HIP_SAFE_CALL(hipMemcpy(mat.data, gmat.getDevicePtr(), mat.total() * mat.elemSize(), hipMemcpyDeviceToHost));
}

void HipMatND::download(OutputArray dst, Stream& stream) const
{
    CV_DbgAssert(!empty());

    dst.create(dims, size.data(), type());
    Mat mat = dst.getMat();

    HipMatND gmat = *this;

    if (!gmat.isContinuous())
        gmat = gmat.clone(stream);

    hipStream_t _stream = StreamAccessor::getStream(stream);
    CV_HIP_SAFE_CALL(hipMemcpyAsync(mat.data, gmat.getDevicePtr(), mat.total() * mat.elemSize(), hipMemcpyDeviceToHost, _stream));
}

void HipMatND::fit(const MatShape& _shape, int _type)
{
    {
        auto elements_nonzero = [](const MatShape& v)
        {
            return std::all_of(v.begin(), v.end(), [](int u){ return u > 0; });
        };
        CV_Assert(!_shape.empty());
        CV_Assert(elements_nonzero(_shape));
    }

    _type &= Mat::TYPE_MASK;

    const size_t esz = (size_t)CV_ELEM_SIZE(_type);
    size_t step0 = esz;
    for (int i = (int)_shape.size() - 2; i >= 0; --i)
        step0 *= (size_t)_shape[(size_t)i + 1];
    const size_t requiredBytes = (size_t)_shape[0] * step0;

    const bool canReuse = !empty() && !external() && isContinuous() && !isSubmatrix() && offset == 0;
    const size_t oldCapacity = canReuse ? totalMemSize() : 0;

    if (!canReuse || requiredBytes > oldCapacity)
    {
        create(_shape, _type);
        return;
    }

    setFields(_shape, _type);
}
