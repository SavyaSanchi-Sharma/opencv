// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Imgproc operations of the HIP backend. Each picks its kernel from the rules in opencv_hip_tuning/
// next to the plugin and from facts about the compiled kernel; nothing is timed.

#ifdef HAVE_ACCEL_HIP

#include "accel/hip/hip.hpp"

#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "kernels.hpp"

#include <dlfcn.h>

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cv { namespace accel { namespace hip_impl {

namespace {

// OPENCV_LOG_LEVEL=INFO logs where the rules came from and each distinct choice, once
bool tracing()
{
    return utils::logging::getLogLevel() >= utils::logging::LOG_LEVEL_INFO;
}

void trace(const std::string& message)
{
    static std::mutex mtx;
    static std::set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!seen.insert(message).second)
            return;
    }
    CV_LOG_INFO(NULL, "accel(hip): " << message);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// rules

struct OpRules
{
    std::vector<std::string> kernels;   //!< best first
    Size block;                         //!< empty when the rules give none
};

struct ArchRules
{
    ArchRules() : minResidentBlocks(1), measuredRuntime(0) {}
    std::string section;                //!< the section these came from, for the trace
    int minResidentBlocks;
    std::map<std::string, OpRules> ops;
    std::string measuredDevice;         //!< set in a section written by opencv_hip_tune
    int measuredRuntime;
};

// The kernels the code has per operation; each is compiled for every architecture of the build.
// Which ones an architecture uses, and in what order, is its section of the rules.
bool implemented(const std::string& op, const std::string& kernel)
{
    static const char* const kernels[][2] = {
        { "boxFilter", "lds_tile" },
        { "boxFilter", "sliding_column" },
        { "nv12ToBgr", "chroma_share" },
    };
    for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++)
        if (op == kernels[i][0] && kernel == kernels[i][1])
            return true;
    return false;
}

Size readBlock(const FileNode& n, const std::string& where)
{
    const Size b = n.size() == 2 ? Size((int)n[0], (int)n[1]) : Size();
    if (b.width <= 0 || b.height <= 0 || b.area() > 256)
        CV_Error_(Error::StsParseError, ("%s: a block is [x, y] with at most 256 threads", where.c_str()));
    return b;
}

ArchRules readArch(const FileNode& node, const std::string& path)
{
    ArchRules a;
    a.section = node.name();
    const std::string where = path + ", " + a.section;
    if (!node["min_resident_blocks"].empty())
        a.minResidentBlocks = (int)node["min_resident_blocks"];
    const FileNode measured = node["measured_on"];
    if (!measured.empty())
    {
        a.measuredDevice = (std::string)measured["device"];
        a.measuredRuntime = (int)measured["hip_runtime"];
    }
    for (FileNodeIterator it = node.begin(); it != node.end(); ++it)
    {
        const FileNode op = *it;
        const std::string name = op.name();
        if (!op.isMap() || name == "measured_on")
            continue;
        OpRules& r = a.ops[name];
        const FileNode ks = op["kernels"];
        for (FileNodeIterator k = ks.begin(); k != ks.end(); ++k)
        {
            const std::string kernel = (std::string)*k;
            if (!implemented(name, kernel))
                CV_Error_(Error::StsParseError, ("%s: %s has no kernel '%s'", where.c_str(), name.c_str(), kernel.c_str()));
            r.kernels.push_back(kernel);
        }
        if (!op["block"].empty())
            r.block = readBlock(op["block"], where);
    }
    return a;
}

// a rules file that cannot be read or parsed throws cv::Exception
std::map<std::string, ArchRules> loadRules(const std::string& path)
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
        CV_Error_(Error::StsError, ("cannot open %s", path.c_str()));
    if ((int)fs["schema"] != 2)
        CV_Error_(Error::StsParseError, ("%s: expected schema 2", path.c_str()));
    std::map<std::string, ArchRules> rules;
    const FileNode archs = fs["archs"];
    for (FileNodeIterator it = archs.begin(); it != archs.end(); ++it)
        rules[(*it).name()] = readArch(*it, path);
    if (tracing())
        trace("kernel rules from " + path);
    return rules;
}

//! the directory of this plugin, with a trailing slash, or empty
std::string pluginDir()
{
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&pluginDir), &info) || !info.dli_fname)
        return std::string();
    const std::string p(info.dli_fname);
    const size_t slash = p.rfind('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash + 1);
}

// imgproc.yaml is shipped, imgproc.tuned.yaml is written by opencv_hip_tune for this machine;
// OPENCV_ACCEL_HIP_RULES=<file> reads that file instead of both
struct RuleFiles
{
    std::map<std::string, ArchRules> shipped, tuned;
};

const RuleFiles& ruleFiles()
{
    static const RuleFiles files = []() {
        RuleFiles f;
        const std::string path = utils::getConfigurationParameterString("OPENCV_ACCEL_HIP_RULES");
        if (!path.empty())
        {
            f.shipped = loadRules(path);
            return f;
        }
        const std::string dir = pluginDir() + "opencv_hip_tuning/";
        f.shipped = loadRules(dir + "imgproc.yaml");
        if (utils::fs::exists(dir + "imgproc.tuned.yaml"))
            f.tuned = loadRules(dir + "imgproc.tuned.yaml");
        return f;
    }();
    return files;
}

const ArchRules* findSection(const std::map<std::string, ArchRules>& rules, const std::string& name)
{
    std::map<std::string, ArchRules>::const_iterator it = rules.find(name);
    return it == rules.end() ? 0 : &it->second;
}

struct DeviceFacts
{
    std::string name;       //!< hipDeviceProp_t::name
    std::string arch;       //!< "gfx1100", without the feature suffix of gcnArchName
    size_t ldsPerBlock;
    int runtime;            //!< hipRuntimeGetVersion
};

const DeviceFacts& deviceFacts(int device)
{
    static std::mutex mtx;
    static std::map<int, DeviceFacts> facts;
    std::lock_guard<std::mutex> lock(mtx);
    std::map<int, DeviceFacts>::iterator it = facts.find(device);
    if (it == facts.end())
    {
        hipDeviceProp_t p;
        check(hipGetDeviceProperties(&p, device), "hipGetDeviceProperties");
        DeviceFacts f;
        f.name = p.name;
        f.arch = p.gcnArchName;
        f.arch = f.arch.substr(0, f.arch.find(':'));
        f.ldsPerBlock = p.sharedMemPerBlock;
        check(hipRuntimeGetVersion(&f.runtime), "hipRuntimeGetVersion");
        it = facts.insert(std::make_pair(device, f)).first;
    }
    return it->second;
}

// The shipped section of the device's architecture (else "default"), with its tuned operations
// on top when they were measured on the same GPU model and HIP runtime. NULL when there are none.
const ArchRules* rulesFor(int device)
{
    static std::mutex mtx;
    static std::map<int, std::pair<bool, ArchRules> > merged;
    const DeviceFacts& dev = deviceFacts(device);
    std::lock_guard<std::mutex> lock(mtx);
    std::map<int, std::pair<bool, ArchRules> >::iterator it = merged.find(device);
    if (it != merged.end())
        return it->second.first ? &it->second.second : 0;

    const RuleFiles& files = ruleFiles();
    const ArchRules* base = findSection(files.shipped, dev.arch);
    if (!base)
        base = findSection(files.shipped, "default");
    const ArchRules* tuned = findSection(files.tuned, dev.arch);
    if (tuned && (tuned->measuredDevice != dev.name || tuned->measuredRuntime != dev.runtime))
    {
        CV_LOG_WARNING(NULL, "accel(hip): the tuned rules for " << dev.arch << " were measured on "
                       << tuned->measuredDevice << " with HIP " << tuned->measuredRuntime << ", this is "
                       << dev.name << " with HIP " << dev.runtime << "; using the shipped rules. "
                       "Run opencv_hip_tune again.");
        tuned = 0;
    }

    ArchRules r;
    if (base)
        r = *base;
    if (tuned)
    {
        for (std::map<std::string, OpRules>::const_iterator op = tuned->ops.begin(); op != tuned->ops.end(); ++op)
            r.ops[op->first] = op->second;
        r.section = tuned->section + " (tuned)" + (base ? " over " + base->section : std::string());
    }
    const bool found = base || tuned;
    it = merged.insert(std::make_pair(device, std::make_pair(found, r))).first;
    return found ? &it->second.second : 0;
}

const OpRules* opRules(const ArchRules* arch, const char* op)
{
    if (!arch)
        return 0;
    std::map<std::string, OpRules>::const_iterator it = arch->ops.find(op);
    return it == arch->ops.end() ? 0 : &it->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// choosing a kernel

// Why @p kernel, as compiled for @p device (the current one), cannot run well there, or NULL.
// One LDS kernel measured 4.5-5x slower with a toolchain that spilled it.
const char* kernelRejected(const void* kernel, int blockThreads, int minResidentBlocks, int device)
{
    static std::mutex mtx;
    static std::map<std::tuple<const void*, int, int, int>, const char*> verdicts;
    const size_t lds = deviceFacts(device).ldsPerBlock;
    std::lock_guard<std::mutex> lock(mtx);
    const std::tuple<const void*, int, int, int> key(kernel, blockThreads, minResidentBlocks, device);
    std::map<std::tuple<const void*, int, int, int>, const char*>::iterator it = verdicts.find(key);
    if (it != verdicts.end())
        return it->second;

    const char* why = 0;
    hipFuncAttributes a;
    int blocks = 0;
    if (hipFuncGetAttributes(&a, kernel) != hipSuccess)
        why = "it is not built for this architecture";
    else if (a.sharedSizeBytes > lds)
        why = "its LDS does not fit";
    else if (a.localSizeBytes > 0)
        why = "it spills to scratch";
    else if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, blockThreads, 0) != hipSuccess)
        why = "its occupancy cannot be read";
    else if (blocks < minResidentBlocks)
        why = "too few of its blocks stay resident";
    if (why)
        (void)hipGetLastError();
    verdicts[key] = why;
    return why;
}

struct Candidate
{
    std::string name;
    const void* kernel;             //!< NULL when the kernel does not apply to the call
    const char* notApplicable;
};

// the first candidate that applies and is not rejected, or -1; @p reasons says why others were skipped
int choose(const std::vector<Candidate>& c, int blockThreads, const ArchRules& arch, int device,
           std::string& reasons)
{
    for (size_t i = 0; i < c.size(); i++)
    {
        const char* why = c[i].kernel ? kernelRejected(c[i].kernel, blockThreads, arch.minResidentBlocks, device)
                                      : c[i].notApplicable;
        if (!why)
            return (int)i;
        reasons += (reasons.empty() ? "" : "; ") + c[i].name + ": " + why;
    }
    return -1;
}

std::string traceTail(const ArchRules* arch, const std::string& reasons)
{
    std::string s = " [rules: " + (arch ? arch->section : std::string("none")) + "]";
    return reasons.empty() ? s : s + " (" + reasons + ")";
}

////////////////////////////////////////////////////////////////////////////////////////////////

class HipImgprocOps CV_FINAL : public ImgprocOps
{
public:
    bool boxFilter(const UMat& src, UMat& dst, Size ksize, Point anchor, bool normalize,
                   int borderType, Stream& stream) CV_OVERRIDE
    {
        const int cn = src.channels();
        if (src.dims != 2 || src.depth() != CV_8U || (cn != 1 && cn != 3 && cn != 4) ||
            dst.type() != src.type() || src.rows >= kernels::IMGPROC_MAX_ROWS)
            return false;
        // invalid anchors are reported by the CPU path
        if (anchor.x < 0 || anchor.x >= ksize.width || anchor.y < 0 || anchor.y >= ksize.height)
            return false;
        // BORDER_WRAP and anything else the kernels do not implement
        const void* sliding = kernels::boxSlidingKernelFor(cn, borderType);
        if (!sliding)
            return false;

        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        const DeviceFacts& dev = deviceFacts(s.device_);
        const ArchRules* arch = rulesFor(s.device_);
        const OpRules* rules = opRules(arch, "boxFilter");

        const void* lds = ksize.width == ksize.height ? kernels::boxLdsKernelFor(cn, ksize.width, borderType) : 0;
        std::vector<Candidate> c;
        for (size_t i = 0; rules && i < rules->kernels.size(); i++)
        {
            if (rules->kernels[i] == "lds_tile")
                c.push_back({ rules->kernels[i], lds, "built for square 3x3 and 5x5 only" });
            else if (rules->kernels[i] == "sliding_column")
                c.push_back({ rules->kernels[i], sliding, 0 });
        }

        std::string reasons;
        const int k = arch ? choose(c, kernels::BOX_BLOCK_THREADS, *arch, s.device_, reasons) : -1;
        if (tracing())
            trace(format("boxFilter %dx%d C%d on %s: ", ksize.width, ksize.height, cn, dev.arch.c_str()) +
                  (k < 0 ? std::string("CPU") : c[k].name) + traceTail(arch, reasons));
        if (k < 0)
            return false;

        const float scale = normalize ? 1.f / ((float)ksize.width * ksize.height) : 1.f;
        const unsigned char* sp = (const unsigned char*)devicePtr(src);
        unsigned char* dp = (unsigned char*)devicePtr(dst);
        if (lds && c[k].kernel == lds)
            check(kernels::boxFilterLds(lds, sp, src.step[0], dp, dst.step[0], src.rows, src.cols,
                                        anchor.x, anchor.y, scale, s.s_), "boxFilter");
        else
            check(kernels::boxFilterSliding(sliding, sp, src.step[0], dp, dst.step[0], src.rows, src.cols,
                                            ksize.width, ksize.height, anchor.x, anchor.y, scale, s.s_), "boxFilter");
        return true;
    }

    bool cvtTwoPlaneYUVtoBGR(const UMat& src, UMat& dst, int dcn, bool swapBlue, int uIdx,
                             Stream& stream) CV_OVERRIDE
    {
        if (src.dims != 2 || src.type() != CV_8UC1 || (dcn != 3 && dcn != 4) ||
            dst.type() != CV_MAKETYPE(CV_8U, dcn))
            return false;
        if (dst.rows % 2 != 0 || dst.cols % 2 != 0 || src.cols != dst.cols ||
            src.rows != dst.rows * 3 / 2 || dst.rows >= kernels::IMGPROC_MAX_ROWS)
            return false;

        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        const DeviceFacts& dev = deviceFacts(s.device_);
        const ArchRules* arch = rulesFor(s.device_);
        const OpRules* rules = opRules(arch, "nv12ToBgr");

        std::vector<Candidate> c;
        for (size_t i = 0; rules && i < rules->kernels.size(); i++)
            if (rules->kernels[i] == "chroma_share")
                c.push_back({ rules->kernels[i], kernels::nv12ToBgrKernelFor(dcn, swapBlue, uIdx), 0 });

        // 64x4 when the rules give no block
        const Size block = rules && !rules->block.empty() ? rules->block : Size(64, 4);
        std::string reasons;
        const int k = arch ? choose(c, block.area(), *arch, s.device_, reasons) : -1;
        if (tracing())
            trace(format("cvtColor NV12 to %d channels on %s: ", dcn, dev.arch.c_str()) +
                  (k < 0 ? std::string("CPU") : format("%s %dx%d", c[k].name.c_str(), block.width, block.height)) +
                  traceTail(arch, reasons));
        if (k < 0)
            return false;

        check(kernels::nv12ToBgr(c[k].kernel, (const unsigned char*)devicePtr(src), src.step[0],
                                 (unsigned char*)devicePtr(dst), dst.step[0], dst.rows, dst.cols,
                                 block.width, block.height, s.s_), "cvtColor");
        return true;
    }
};

} // namespace

ImgprocOps* imgprocOps()
{
    static HipImgprocOps ops;
    return &ops;
}

}}} // namespace cv::accel::hip_impl

#endif // HAVE_ACCEL_HIP
