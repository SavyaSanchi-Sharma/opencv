// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Times the HIP imgproc kernels on this machine's GPUs and writes the fastest choice per GPU
// architecture to opencv_hip_tuning/imgproc.tuned.yaml, which the HIP plugin reads at startup.

#include <opencv2/core.hpp>
#include <opencv2/core/accel.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace cv;

namespace {

const char* const USAGE =
    "usage: opencv_hip_tune [--device hip:N] [--dry-run] [--output FILE] [--reset]\n"
    "  --device hip:N   tune one GPU (default: every HIP GPU)\n"
    "  --dry-run        log the choices, write nothing\n"
    "  --output FILE    write FILE instead of opencv_hip_tuning/imgproc.tuned.yaml\n"
    "  --reset          delete the tuned file: back to the shipped rules";

const int REPEATS = 3;          // worker runs per candidate and case
const double MIN_NOISE = 0.02;  // a challenger must beat the shipped choice by more than this

struct Options
{
    bool worker = false, dryRun = false, reset = false;
    std::string device, output;
    std::string op, result;     // worker only
    Size size;
    int k = 0;
};

bool parseSize(const std::string& s, Size& sz)
{
    int w = 0, h = 0;
    if (std::sscanf(s.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0)
        return false;
    sz = Size(w, h);
    return true;
}

std::string sizeName(Size sz)
{
    return format("%dx%d", sz.width, sz.height);
}

// the directory of the OpenCV libraries, where the HIP plugin and its rules are
std::string libDir()
{
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&cv::getBuildInformation), &info) || !info.dli_fname)
        CV_Error(Error::StsError, "cannot find the OpenCV libraries");
    const std::string p(info.dli_fname);
    const size_t slash = p.rfind('/');
    return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

std::string selfPath()
{
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0)
        CV_Error(Error::StsError, "cannot find this executable");
    return std::string(buf, (size_t)n);
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

////////////////////////////////////////////////////////////////////////////////////////////////
// worker: one candidate and one case per process, since the plugin reads its rules once

void writeResult(const std::string& path, bool ok, double us, const std::string& why)
{
    FileStorage fs(path, FileStorage::WRITE | FileStorage::FORMAT_YAML);
    if (!fs.isOpened())
        CV_Error_(Error::StsError, ("cannot write %s", path.c_str()));
    fs << "ok" << (int)ok << "us" << us << "why" << why;
}

void worker(const Options& o)
{
    ocl::setUseOpenCL(false);   // a call the device declines must reach the CPU code
    const accel::Device dev = accel::Device::find(o.device);
    if (dev.empty())
        return writeResult(o.result, false, 0, "no device " + o.device);
    accel::ExecutionScope scope(dev);

    RNG rng(0x5eed);
    Mat ref;
    UMat usrc, udst;
    std::function<void()> run;
    if (o.op == "boxFilter")
    {
        Mat src(o.size, CV_8UC3);
        rng.fill(src, RNG::UNIFORM, 0, 256);
        boxFilter(src, ref, -1, Size(o.k, o.k));
        src.copyTo(usrc);
        run = [&]() { boxFilter(usrc, udst, -1, Size(o.k, o.k)); };
    }
    else if (o.op == "nv12ToBgr")
    {
        Mat yuv(o.size.height * 3 / 2, o.size.width, CV_8UC1);
        rng.fill(yuv, RNG::UNIFORM, 0, 256);
        cvtColor(yuv, ref, COLOR_YUV2BGR_NV12);
        yuv.copyTo(usrc);
        run = [&]() { cvtColor(usrc, udst, COLOR_YUV2BGR_NV12); };
    }
    else
        CV_Error_(Error::StsBadArg, ("unknown operation %s", o.op.c_str()));

    accel::resetTransferStats();
    run();
    if (accel::getTransferStats().deviceOps == 0)
        return writeResult(o.result, false, 0, "not run on this GPU (OPENCV_LOG_LEVEL=INFO shows why)");
    Mat got = udst.getMat(ACCESS_READ).clone(), diff;
    absdiff(ref, got, diff);
    if (countNonZero(diff.reshape(1)) != 0)
        return writeResult(o.result, false, 0, "differs from the CPU result");

    for (int i = 0; i < 20; i++)
        run();
    accel::synchronize(dev);
    const int calls = 50;
    std::vector<double> rounds;
    for (int r = 0; r < 7; r++)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < calls; i++)
            run();
        accel::synchronize(dev);
        rounds.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / calls);
    }
    writeResult(o.result, true, median(rounds), "");
}

////////////////////////////////////////////////////////////////////////////////////////////////
// driver

struct Run
{
    bool ok;
    double us;
    std::string why;
};

// times @p kernel (with @p block, if any) for @p op in a worker that sees only that candidate
Run runWorker(const std::string& self, const std::string& device, const std::string& op,
              const std::string& kernel, Size block, Size size, int k)
{
    const std::string rules = tempfile(".yaml"), result = tempfile(".yaml");
    {
        std::ofstream f(rules.c_str());
        f << "%YAML:1.0\n---\nschema: 2\narchs:\n   default:\n      min_resident_blocks: 1\n"
          << "      " << op << ":\n         kernels: [ " << kernel << " ]\n";
        if (!block.empty())
            f << "         block: [ " << block.width << ", " << block.height << " ]\n";
    }
    std::string cmd = format("OPENCV_ACCEL_HIP_RULES='%s' OPENCV_LOG_LEVEL=ERROR '%s' --worker --device %s "
                             "--op %s --size %s --result '%s'", rules.c_str(), self.c_str(), device.c_str(),
                             op.c_str(), sizeName(size).c_str(), result.c_str());
    if (k)
        cmd += format(" --k %d", k);
    const int status = std::system((cmd + " >/dev/null 2>&1").c_str());

    Run r = { false, 0, format("the worker failed (status %d)", status) };
    FileStorage fs(result, FileStorage::READ);
    if (fs.isOpened())
        r = { (int)fs["ok"] != 0, (double)fs["us"], (std::string)fs["why"] };
    fs.release();
    std::remove(rules.c_str());
    std::remove(result.c_str());
    return r;
}

struct Case
{
    std::string label;
    Size size;
    int k;
};

struct Choice
{
    std::vector<std::string> order;     //!< winner first
    std::string evidence;
};

// Times every candidate on every case REPEATS times, rotating the order, and keeps @p incumbent
// unless another is faster (geometric mean over the cases) by more than the spread of the repeats.
Choice decide(const std::vector<std::string>& candidates, const std::string& incumbent,
              const std::vector<Case>& cases,
              const std::function<Run(const std::string&, const Case&)>& measure)
{
    const size_t n = candidates.size();
    std::map<std::string, std::vector<std::vector<double> > > times;
    std::map<std::string, std::string> failed;
    for (int rep = 0; rep < REPEATS; rep++)
        for (size_t ci = 0; ci < cases.size(); ci++)
            for (size_t j = 0; j < n; j++)
            {
                const std::string& c = candidates[(j + rep) % n];
                if (failed.count(c))
                    continue;
                const Run r = measure(c, cases[ci]);
                if (!r.ok)
                {
                    failed[c] = r.why;
                    CV_LOG_INFO(NULL, "    " << c << "  " << cases[ci].label << "  failed: " << r.why);
                    continue;
                }
                times[c].resize(cases.size());
                times[c][ci].push_back(r.us);
                CV_LOG_INFO(NULL, format("    %-16s %-8s %10.1f us", c.c_str(), cases[ci].label.c_str(), r.us));
            }

    std::map<std::string, double> score;
    double noise = MIN_NOISE;
    std::vector<std::string> ranked;
    for (size_t j = 0; j < n; j++)
    {
        const std::string& c = candidates[j];
        if (failed.count(c))
            continue;
        double logSum = 0;
        for (size_t ci = 0; ci < cases.size(); ci++)
        {
            const std::vector<double>& t = times[c][ci];
            const double m = median(t);
            logSum += std::log(m);
            noise = std::max(noise, (*std::max_element(t.begin(), t.end()) - *std::min_element(t.begin(), t.end())) / m);
        }
        score[c] = std::exp(logSum / cases.size());
        ranked.push_back(c);
    }
    std::sort(ranked.begin(), ranked.end(), [&](const std::string& a, const std::string& b) { return score[a] < score[b]; });

    std::string winner = ranked.empty() ? incumbent : ranked[0];
    if (!ranked.empty() && winner != incumbent && score.count(incumbent) &&
        score[incumbent] / score[winner] - 1 <= noise)
        winner = incumbent;     // inside the noise: keep the shipped choice

    Choice ch;
    ch.order.push_back(winner);
    for (size_t j = 0; j < ranked.size(); j++)
        if (ranked[j] != winner)
            ch.order.push_back(ranked[j]);
    for (size_t j = 0; j < n; j++)
        if (failed.count(candidates[j]) && candidates[j] != winner)
            ch.order.push_back(candidates[j]);  // the plugin still checks it on every call
    for (size_t ci = 0; ci < cases.size(); ci++)
    {
        ch.evidence += (ci ? "; " : "") + cases[ci].label + ":";
        for (size_t j = 0; j < ranked.size(); j++)
            ch.evidence += format(" %s %.1f us", ranked[j].c_str(), median(times[ranked[j]][ci]));
    }
    ch.evidence += format("; spread %.1f%%", noise * 100);
    return ch;
}

struct OpOut
{
    std::vector<std::string> kernels;
    Size block;
    std::string evidence;
};

struct Section
{
    std::string arch, device, date;
    int runtime = 0;
    std::map<std::string, OpOut> ops;
};

// the sections already in the tuned file, so that GPUs not tuned now keep theirs
std::map<std::string, Section> readTuned(const std::string& path)
{
    std::map<std::string, Section> out;
    if (!utils::fs::exists(path))
        return out;
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
        CV_Error_(Error::StsError, ("cannot open %s", path.c_str()));
    const FileNode archs = fs["archs"];
    for (FileNodeIterator it = archs.begin(); it != archs.end(); ++it)
    {
        Section s;
        s.arch = (*it).name();
        const FileNode m = (*it)["measured_on"];
        s.device = (std::string)m["device"];
        s.runtime = (int)m["hip_runtime"];
        s.date = (std::string)m["date"];
        for (FileNodeIterator o = (*it).begin(); o != (*it).end(); ++o)
        {
            if (!(*o).isMap() || (*o).name() == "measured_on")
                continue;
            OpOut& op = s.ops[(*o).name()];
            const FileNode ks = (*o)["kernels"];
            for (FileNodeIterator k = ks.begin(); k != ks.end(); ++k)
                op.kernels.push_back((std::string)*k);
            const FileNode b = (*o)["block"];
            if (b.size() == 2)
                op.block = Size((int)b[0], (int)b[1]);
            op.evidence = (std::string)(*o)["evidence"];
        }
        out[s.arch] = s;
    }
    return out;
}

void writeTuned(const std::string& path, const std::map<std::string, Section>& sections)
{
    const size_t slash = path.rfind('/');
    if (slash != std::string::npos)
        utils::fs::createDirectories(path.substr(0, slash));
    const std::string tmp = path + ".tmp";
    {
        FileStorage fs(tmp, FileStorage::WRITE | FileStorage::FORMAT_YAML);
        if (!fs.isOpened())
            CV_Error_(Error::StsError, ("cannot write %s", tmp.c_str()));
        fs.writeComment(" Written by opencv_hip_tune; used ahead of imgproc.yaml for the GPU and HIP runtime");
        fs.writeComment(" each section was measured on. opencv_hip_tune --reset deletes it.");
        fs << "schema" << 2 << "archs" << "{";
        for (std::map<std::string, Section>::const_iterator it = sections.begin(); it != sections.end(); ++it)
        {
            const Section& s = it->second;
            fs << s.arch << "{";
            fs << "measured_on" << "{" << "device" << s.device << "hip_runtime" << s.runtime << "date" << s.date << "}";
            for (std::map<std::string, OpOut>::const_iterator op = s.ops.begin(); op != s.ops.end(); ++op)
            {
                fs << op->first << "{" << "kernels" << "[:";
                for (size_t k = 0; k < op->second.kernels.size(); k++)
                    fs << op->second.kernels[k];
                fs << "]";
                if (!op->second.block.empty())
                    fs << "block" << "[:" << op->second.block.width << op->second.block.height << "]";
                if (!op->second.evidence.empty())
                    fs << "evidence" << op->second.evidence;
                fs << "}";
            }
            fs << "}";
        }
        fs << "}";
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        CV_Error_(Error::StsError, ("cannot replace %s", path.c_str()));
}

// the shipped section for @p arch (else "default"): boxFilter's first kernel and nv12ToBgr's block
void readShipped(const std::string& path, const std::string& arch, std::string& section,
                 std::string& boxFirst, Size& nvBlock)
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
        CV_Error_(Error::StsError, ("cannot open the shipped rules %s", path.c_str()));
    const FileNode archs = fs["archs"];
    FileNode a = archs[arch];
    section = arch;
    if (a.empty())
    {
        a = archs["default"];
        section = "default";
    }
    if (a.empty())
        CV_Error_(Error::StsError, ("%s has no section for %s and no default", path.c_str(), arch.c_str()));
    const FileNode ks = a["boxFilter"]["kernels"];
    boxFirst = ks.size() > 0 ? (std::string)ks[0] : std::string("lds_tile");
    const FileNode b = a["nv12ToBgr"]["block"];
    nvBlock = b.size() == 2 ? Size((int)b[0], (int)b[1]) : Size(64, 4);
}

void hipCheck(hipError_t e, const char* what)
{
    if (e != hipSuccess)
        CV_Error_(Error::GpuApiCallError, ("%s failed: %s", what, hipGetErrorString(e)));
}

std::string joined(const std::vector<std::string>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); i++)
        s += (i ? ", " : "") + v[i];
    return s;
}

void tune(const Options& o)
{
    const std::string dir = libDir() + "/opencv_hip_tuning";
    const std::string out = o.output.empty() ? dir + "/imgproc.tuned.yaml" : o.output;
    if (o.reset)
    {
        if (utils::fs::exists(out) && std::remove(out.c_str()) != 0)
            CV_Error_(Error::StsError, ("cannot delete %s", out.c_str()));
        CV_LOG_INFO(NULL, "deleted " << out << ": the shipped rules apply from the next program start");
        return;
    }

    std::vector<accel::Device> devs;
    for (const accel::Device& d : accel::Device::enumerate())
        if (d.backend() == "hip" && (o.device.empty() || d.name() == o.device))
            devs.push_back(d);
    if (devs.empty())
        CV_Error_(Error::StsError, ("no HIP GPU %s: is libopencv_core_accel_hip.so in %s?",
                                    o.device.c_str(), libDir().c_str()));

    const std::string self = selfPath();
    std::map<std::string, Section> sections = readTuned(out);
    int runtime = 0;
    hipCheck(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
    char date[16];
    const std::time_t now = std::time(0);
    std::strftime(date, sizeof(date), "%Y-%m-%d", std::localtime(&now));

    for (const accel::Device& dev : devs)
    {
        hipDeviceProp_t p;
        hipCheck(hipGetDeviceProperties(&p, dev.index()), "hipGetDeviceProperties");
        std::string arch = p.gcnArchName;
        arch = arch.substr(0, arch.find(':'));
        std::string shippedSection, boxFirst;
        Size nvBlock;
        readShipped(dir + "/imgproc.yaml", arch, shippedSection, boxFirst, nvBlock);
        CV_LOG_INFO(NULL, dev.name() << "  " << p.name << "  " << arch << "  (shipped rules: " << shippedSection << ")");

        Section s;
        s.arch = arch;
        s.device = p.name;
        s.runtime = runtime;
        s.date = date;

        // the LDS tile exists for 3x3 and 5x5 only; larger windows have nothing to choose from
        CV_LOG_INFO(NULL, "  boxFilter, 8UC3");
        const std::vector<Case> boxCases = { { "FHD k3", Size(1920, 1080), 3 }, { "FHD k5", Size(1920, 1080), 5 },
                                             { "4K k5", Size(3840, 2160), 5 } };
        const Choice box = decide({ "lds_tile", "sliding_column" }, boxFirst, boxCases,
            [&](const std::string& kernel, const Case& c) {
                return runWorker(self, dev.name(), "boxFilter", kernel, Size(), c.size, c.k);
            });
        s.ops["boxFilter"] = { box.order, Size(), box.evidence };

        // the block shape reaches the kernel at launch, so these need no rebuild
        CV_LOG_INFO(NULL, "  nv12ToBgr, NV12 to BGR");
        const std::vector<Case> nvCases = { { "FHD", Size(1920, 1080), 0 }, { "4K", Size(3840, 2160), 0 } };
        const Choice nv = decide({ "64x4", "32x8", "128x2", "256x1" }, sizeName(nvBlock), nvCases,
            [&](const std::string& block, const Case& c) {
                Size b;
                parseSize(block, b);
                return runWorker(self, dev.name(), "nv12ToBgr", "chroma_share", b, c.size, 0);
            });
        Size best;
        parseSize(nv.order[0], best);
        s.ops["nv12ToBgr"] = { { "chroma_share" }, best, nv.evidence };

        CV_LOG_INFO(NULL, "  -> " << arch << ": boxFilter [" << joined(box.order) << "], nv12ToBgr block " << sizeName(best));
        if (sections.count(arch) && sections[arch].device != s.device)
            CV_LOG_INFO(NULL, "  (replaces the section measured on " << sections[arch].device << ")");
        sections[arch] = s;
    }

    if (o.dryRun)
    {
        CV_LOG_INFO(NULL, "dry run: nothing written");
        return;
    }
    writeTuned(out, sections);
    CV_LOG_INFO(NULL, "wrote " << out << "; programs started from now on use it, --reset goes back to the shipped rules");
}

Options parseArgs(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (a == "--worker") o.worker = true;
        else if (a == "--dry-run") o.dryRun = true;
        else if (a == "--reset") o.reset = true;
        else if (a == "--device" && hasValue) o.device = argv[++i];
        else if (a == "--output" && hasValue) o.output = argv[++i];
        else if (a == "--op" && hasValue) o.op = argv[++i];
        else if (a == "--result" && hasValue) o.result = argv[++i];
        else if (a == "--k" && hasValue) o.k = std::atoi(argv[++i]);
        else if (a == "--size" && hasValue && parseSize(argv[++i], o.size)) continue;
        else CV_Error_(Error::StsBadArg, ("%s", USAGE));
    }
    if (o.worker && (o.result.empty() || o.op.empty() || o.size.empty()))
        CV_Error(Error::StsBadArg, "--worker needs --op, --size and --result");
    return o;
}

} // namespace

int main(int argc, char** argv)
{
    // the progress is logged at INFO, which a Release build hides unless asked otherwise
    if (utils::getConfigurationParameterString("OPENCV_LOG_LEVEL").empty())
        utils::logging::setLogLevel(utils::logging::LOG_LEVEL_INFO);
    const Options o = parseArgs(argc, argv);
    if (o.worker)
        worker(o);
    else
        tune(o);
    return 0;
}
