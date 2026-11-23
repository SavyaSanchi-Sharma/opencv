// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the HIP backend. Memory, streams, flip and transpose live in gpu/; setTo,
// convertTo and the imgproc operations in hal/accel/hip.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#ifdef HAVE_HIP

#define CV_ACCEL_GPU_HIP
#include "gpu/gpu_backend.impl.hpp"
#include "hip/hip.hpp"

namespace cv { namespace accel { namespace hip_impl {

void hipCheck(hipError_t e, const char* what)
{
    check(e, what);
}

HipDeviceGuard::HipDeviceGuard(int device) : prev_(-1)
{
    int current = -1;
    hipCheck(hipGetDevice(&current), "hipGetDevice");
    if (current != device)
    {
        hipCheck(hipSetDevice(device), "hipSetDevice");
        prev_ = current;
    }
}

HipDeviceGuard::~HipDeviceGuard()
{
    if (prev_ >= 0)
        (void)hipSetDevice(prev_);
}

HipStreamRef hipStreamOf(Stream& stream)
{
    GpuStream& s = gpuStreamOf(stream);
    HipStreamRef r = { s.device_, s.s_ };
    return r;
}

namespace {

// flip and transpose are the kernels shared with CUDA, setTo and convertTo those of hal/accel/hip
class HipBackendCoreOps CV_FINAL : public CoreOps
{
public:
    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE
    {
        return gpu_.flip(src, dst, flipCode, stream);
    }
    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE
    {
        return gpu_.transpose(src, dst, stream);
    }
    bool setTo(UMat& dst, const Scalar& value, Stream& stream) CV_OVERRIDE
    {
        return coreOps()->setTo(dst, value, stream);
    }
    bool convertTo(const UMat& src, UMat& dst, double alpha, double beta, Stream& stream) CV_OVERRIDE
    {
        return coreOps()->convertTo(src, dst, alpha, beta, stream);
    }

private:
    GpuCoreOps gpu_;
};

class HipBackend CV_FINAL : public GpuBackend
{
public:
    HipBackend()
    {
        for (size_t i = 0; i < devs_.size(); i++)
        {
            GpuDevice& d = *devs_[i];
            d.shared = d.prop.integrated && d.prop.canMapHostMemory;
        }
    }

    String name() const CV_OVERRIDE { return "hip"; }

    Ops* ops(const char* module) CV_OVERRIDE
    {
        if (strcmp(module, "core") == 0)
            return &ops_;
        if (strcmp(module, "imgproc") == 0)
            return imgprocOps();
        return 0;
    }

private:
    HipBackendCoreOps ops_;
};

} // namespace

Ptr<Backend> createHipBackend()
{
    return makePtr<HipBackend>();
}

}}} // namespace cv::accel::hip_impl

#endif // HAVE_HIP
