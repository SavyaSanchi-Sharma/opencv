// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/core/accel.hpp"

namespace opencv_test { namespace {

// HIP imgproc ops against the CPU; OPENCV_ACCEL_HIP_RULES can force sliding_column for 3x3 and 5x5
std::vector<accel::Device> hipDevices()
{
    std::vector<accel::Device> devs;
    for (const accel::Device& d : accel::Device::enumerate())
        if (d.backend() == "hip")
            devs.push_back(d);
    return devs;
}

UMat onDevice(const accel::Device& d, const Mat& m)
{
    UMat u;
    u.allocator = d.allocator();
    u.create(m.size(), m.type());
    u.allocator = 0;
    m.copyTo(u);
    return u;
}

Mat toHost(const UMat& u)
{
    Mat m;
    u.copyTo(m);
    return m;
}

Mat randomMat(RNG& rng, Size size, int type)
{
    Mat m(size, type);
    rng.fill(m, RNG::UNIFORM, 0, 256);
    return m;
}

std::string borderName(int border)
{
    static const char* const names[] = { "CONSTANT", "REPLICATE", "REFLECT", "WRAP", "REFLECT_101" };
    const int b = border & ~BORDER_ISOLATED;
    return std::string(b >= 0 && b <= 4 ? names[b] : "?") + (border & BORDER_ISOLATED ? "|ISOLATED" : "");
}

class Imgproc_AccelHip : public testing::Test
{
protected:
    void SetUp() CV_OVERRIDE
    {
        devs = hipDevices();
        if (devs.empty())
            throw SkipTestException("no HIP device");
        accel::setTransferPolicy(accel::TRANSFER_ALLOW);
    }
    void TearDown() CV_OVERRIDE
    {
        for (size_t i = 0; i < devs.size(); i++)
            accel::synchronize(devs[i]);
    }

    void checkBox(const accel::Device& dev, const Mat& src, Size ksize, Point anchor, bool normalize, int border)
    {
        Mat expected;
        cv::boxFilter(src, expected, -1, ksize, anchor, normalize, border);
        UMat u = onDevice(dev, src), dst;
        accel::resetTransferStats();
        cv::boxFilter(u, dst, -1, ksize, anchor, normalize, border);
        const accel::TransferStats st = accel::getTransferStats();
        const std::string what = cv::format("%s: %dx%d C%d, ksize %dx%d, anchor (%d, %d), %s, ", dev.name().c_str(),
                                            src.cols, src.rows, src.channels(), ksize.width, ksize.height,
                                            anchor.x, anchor.y, normalize ? "normalized" : "sum") + borderName(border);
        EXPECT_EQ(1u, st.deviceOps) << what;
        EXPECT_EQ(dev, accel::getDevice(dst)) << what;
        EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << what;
    }

    std::vector<accel::Device> devs;
};

const int BORDERS[] = { BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101 };

TEST_F(Imgproc_AccelHip, boxFilter_odd_windows_match_the_cpu)
{
    const Size sizes[] = { Size(640, 480), Size(37, 23), Size(3, 2), Size(5, 300) };
    const Size ksizes[] = { Size(3, 3), Size(5, 5), Size(7, 7), Size(3, 5), Size(1, 3), Size(17, 15), Size(31, 31) };
    for (const accel::Device& dev : devs)
    {
        RNG rng(0x5eed);
        for (const Size& sz : sizes)
            for (int cn : { 1, 3, 4 })
            {
                const Mat src = randomMat(rng, sz, CV_8UC(cn));
                for (const Size& k : ksizes)
                    for (int border : BORDERS)
                        checkBox(dev, src, k, Point(-1, -1), true, border);
            }
    }
}

TEST_F(Imgproc_AccelHip, boxFilter_even_windows_match_the_cpu)
{
    const Size ksizes[] = { Size(2, 2), Size(4, 4), Size(2, 3), Size(16, 16), Size(18, 18), Size(20, 20) };
    for (const accel::Device& dev : devs)
    {
        RNG rng(0x5eed);
        for (int cn : { 1, 3, 4 })
        {
            const Mat src = randomMat(rng, Size(128, 96), CV_8UC(cn));
            for (const Size& k : ksizes)
                for (int border : BORDERS)
                    checkBox(dev, src, k, Point(-1, -1), true, border);
        }
    }
}

TEST_F(Imgproc_AccelHip, boxFilter_anchors_and_unnormalized_sums)
{
    const Size ksizes[] = { Size(3, 3), Size(5, 5), Size(7, 7), Size(3, 5), Size(4, 4) };
    for (const accel::Device& dev : devs)
    {
        RNG rng(0x5eed);
        for (int cn : { 1, 3, 4 })
        {
            const Mat src = randomMat(rng, Size(37, 23), CV_8UC(cn));
            for (const Size& k : ksizes)
                for (int border : BORDERS)
                {
                    checkBox(dev, src, k, Point(0, 0), true, border);
                    checkBox(dev, src, k, Point(k.width - 1, k.height - 1), true, border);
                    checkBox(dev, src, k, Point(-1, -1), false, border);
                }
        }
    }
}

TEST_F(Imgproc_AccelHip, boxFilter_region)
{
    RNG rng(0x5eed);
    const Mat big = randomMat(rng, Size(300, 200), CV_8UC3);
    const Rect r(17, 9, 201, 133);
    for (const accel::Device& dev : devs)
    {
        UMat ubig = onDevice(dev, big);
        for (int border : BORDERS)
        {
            for (int isolated : { 0, (int)BORDER_ISOLATED })
            {
                Mat expected;
                cv::boxFilter(big(r), expected, -1, Size(5, 5), Point(-1, -1), true, border | isolated);
                UMat dst;
                accel::resetTransferStats();
                cv::boxFilter(ubig(r), dst, -1, Size(5, 5), Point(-1, -1), true, border | isolated);
                const std::string what = dev.name() + ", " + borderName(border | isolated);
                EXPECT_EQ(isolated ? 1u : 0u, accel::getTransferStats().deviceOps) << what;
                EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << what;
            }
        }
    }
}

TEST_F(Imgproc_AccelHip, boxFilter_unsupported_calls_run_on_the_cpu)
{
    RNG rng(0x5eed);
    const Mat u8c2 = randomMat(rng, Size(40, 30), CV_8UC2), u16 = randomMat(rng, Size(40, 30), CV_16UC1);
    for (const accel::Device& dev : devs)
    {
        struct { const Mat* src; int ddepth; const char* what; } cases[] = {
            { &u8c2, -1, "2 channels" }, { &u16, -1, "16-bit" }, { &u8c2, CV_16U, "8-bit to 16-bit" } };
        for (const auto& c : cases)
        {
            Mat expected;
            cv::boxFilter(*c.src, expected, c.ddepth, Size(3, 3));
            UMat u = onDevice(dev, *c.src), dst;
            accel::resetTransferStats();
            cv::boxFilter(u, dst, c.ddepth, Size(3, 3));
            EXPECT_EQ(0u, accel::getTransferStats().deviceOps) << dev.name() << ", " << c.what;
            EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << dev.name() << ", " << c.what;
        }
    }
}

TEST_F(Imgproc_AccelHip, cvtColor_two_plane_yuv_matches_the_cpu)
{
    const int codes[] = { COLOR_YUV2BGR_NV12, COLOR_YUV2RGB_NV12, COLOR_YUV2BGRA_NV12, COLOR_YUV2RGBA_NV12,
                          COLOR_YUV2BGR_NV21, COLOR_YUV2RGB_NV21, COLOR_YUV2BGRA_NV21, COLOR_YUV2RGBA_NV21 };
    const Size sizes[] = { Size(1920, 1080), Size(34, 22), Size(2, 2), Size(640, 2) };   // of the BGR image
    for (const accel::Device& dev : devs)
    {
        RNG rng(0x5eed);
        for (const Size& sz : sizes)
        {
            const Mat yuv = randomMat(rng, Size(sz.width, sz.height * 3 / 2), CV_8UC1);
            UMat u = onDevice(dev, yuv);
            for (int code : codes)
            {
                Mat expected;
                cv::cvtColor(yuv, expected, code);
                UMat dst;
                accel::resetTransferStats();
                cv::cvtColor(u, dst, code);
                const std::string what = cv::format("%s: %dx%d, code %d", dev.name().c_str(), sz.width, sz.height, code);
                EXPECT_EQ(1u, accel::getTransferStats().deviceOps) << what;
                EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << what;
            }
        }
    }
}

}} // namespace
