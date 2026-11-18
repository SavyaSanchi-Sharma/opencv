// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Times the HIP imgproc kernels on this machine's GPUs and writes their ranking per GPU architecture,
// with workload rules where a workload ranks them differently, to opencv_hip_tuning/imgproc.tuned.yaml.

#include <opencv2/core.hpp>
#include <opencv2/core/accel.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>

#include <hip/hip_runtime_api.h>

#include "imgproc/rules.hpp"

#include <link.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace cv;
namespace rules = cv::accel::hip_impl::rules;

namespace {

const char* const USAGE =
    "usage: opencv_hip_tune [--device hip:N] [--dry-run] [--output FILE] [--reset]\n"
    "       opencv_hip_tune --report [--device hip:N]\n"
    "       opencv_hip_tune --check [FILE]\n"
    "  --device hip:N   tune one GPU (default: every HIP GPU)\n"
    "  --dry-run        log the choices, write nothing\n"
    "  --output FILE    write FILE instead of opencv_hip_tuning/imgproc.tuned.yaml\n"
    "  --reset          delete the tuned file: back to the shipped rules\n"
    "  --report         measure twice and report the regret of the installed and of newly tuned rules\n"
    "                   against the fastest kernel of each case, and whether the tuning is stable\n"
    "  --check [FILE]   validate FILE, else the rules this machine uses; exit status 1 on a problem";

const int REPEATS = 3;          // worker runs per candidate and case
const double MIN_NOISE = 0.02;  // a challenger must beat the incumbent by more than this

struct Options
{
    bool worker = false, dryRun = false, reset = false, report = false, check = false;
    std::string device, output, checkFile;
    std::string op, result;     // worker only
    Size size;
    int cn = 3, k = 0;
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

std::string joined(const std::vector<std::string>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); i++)
        s += (i ? ", " : "") + v[i];
    return s;
}

std::string joined(const std::vector<int>& v)
{
    std::vector<std::string> s;
    for (int x : v)
        s.push_back(std::to_string(x));
    return joined(s);
}

std::string selfPath()
{
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0)
        CV_Error(Error::StsError, "cannot find this executable");
    return std::string(buf, (size_t)n);
}

int findPlugin(dl_phdr_info* info, size_t, void* path)
{
    if (!info->dlpi_name || !std::strstr(info->dlpi_name, "libopencv_core_accel_hip"))
        return 0;
    *static_cast<std::string*>(path) = info->dlpi_name;
    return 1;
}

// the HIP plugin, which accel::Device::enumerate loads; its rules are next to it
std::string pluginPath()
{
    accel::Device::enumerate();
    std::string p;
    dl_iterate_phdr(findPlugin, &p);
    if (p.empty())
        CV_Error(Error::StsError, "the HIP plugin is not loaded: is libopencv_core_accel_hip.so next to the OpenCV libraries?");
    return p;
}

std::string tuningDir()
{
    const std::string p = pluginPath();
    const size_t slash = p.rfind('/');
    return (slash == std::string::npos ? std::string(".") : p.substr(0, slash)) + "/opencv_hip_tuning";
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void logWarnings(const std::vector<std::string>& warnings)
{
    for (const std::string& w : warnings)
        CV_LOG_WARNING(NULL, w);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// worker: one candidate and one case per process, since the plugin reads its rules once

void writeResult(const std::string& path, const std::string& status, double us, const std::string& why)
{
    FileStorage fs(path, FileStorage::WRITE | FileStorage::FORMAT_YAML);
    if (!fs.isOpened())
        CV_Error_(Error::StsError, ("cannot write %s", path.c_str()));
    fs << "status" << status << "us" << us << "why" << why;
}

void worker(const Options& o)
{
    ocl::setUseOpenCL(false);   // a call the device declines must reach the CPU code
    const accel::Device dev = accel::Device::find(o.device);
    if (dev.empty())
        return writeResult(o.result, "failed", 0, "no device " + o.device);
    accel::ExecutionScope scope(dev);

    RNG rng(0x5eed);
    Mat ref;
    UMat usrc, udst;
    std::function<void()> run;
    if (o.op == "boxFilter")
    {
        Mat src(o.size, CV_8UC(o.cn));
        rng.fill(src, RNG::UNIFORM, 0, 256);
        boxFilter(src, ref, -1, Size(o.k, o.k));
        src.copyTo(usrc);
        run = [&]() { boxFilter(usrc, udst, -1, Size(o.k, o.k)); };
    }
    else if (o.op == "nv12ToBgr")
    {
        const int code = o.cn == 4 ? COLOR_YUV2BGRA_NV12 : COLOR_YUV2BGR_NV12;
        Mat yuv(o.size.height * 3 / 2, o.size.width, CV_8UC1);
        rng.fill(yuv, RNG::UNIFORM, 0, 256);
        cvtColor(yuv, ref, code);
        yuv.copyTo(usrc);
        run = [&, code]() { cvtColor(usrc, udst, code); };
    }
    else
        CV_Error_(Error::StsBadArg, ("unknown operation %s", o.op.c_str()));

    accel::resetTransferStats();
    run();
    if (accel::getTransferStats().deviceOps == 0)
        return writeResult(o.result, "declined", 0, "not run on this GPU (OPENCV_LOG_LEVEL=INFO shows why)");
    Mat got = udst.getMat(ACCESS_READ).clone(), diff;
    absdiff(ref, got, diff);
    if (countNonZero(diff.reshape(1)) != 0)
        return writeResult(o.result, "failed", 0, "differs from the CPU result");

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
    writeResult(o.result, "ok", median(rounds), "");
}

////////////////////////////////////////////////////////////////////////////////////////////////
// measuring

struct Run
{
    std::string status;     //!< ok, declined (the plugin did not run it here) or failed
    double us;
    std::string why;
};

struct Case
{
    std::string label;
    Size size;
    int cn, k;
};

// times @p kernel (with @p block, if any) for @p op in a worker that sees only that candidate
Run runWorker(const std::string& self, const std::string& device, const std::string& op,
              const std::string& kernel, Size block, const Case& c)
{
    const std::string rulesPath = tempfile(".yaml"), result = tempfile(".yaml");
    {
        std::ofstream f(rulesPath.c_str());
        f << "%YAML:1.0\n---\nschema: " << rules::SCHEMA << "\narchs:\n   default:\n"
          << "      " << op << ":\n         kernels: [ " << kernel << " ]\n";
        if (!block.empty())
            f << "         block: [ " << block.width << ", " << block.height << " ]\n";
    }
    std::string cmd = format("OPENCV_ACCEL_HIP_RULES='%s' OPENCV_LOG_LEVEL=ERROR '%s' --worker --device %s "
                             "--op %s --size %s --cn %d --result '%s'", rulesPath.c_str(), self.c_str(), device.c_str(),
                             op.c_str(), sizeName(c.size).c_str(), c.cn, result.c_str());
    if (c.k)
        cmd += format(" --k %d", c.k);
    const int status = std::system((cmd + " >/dev/null 2>&1").c_str());

    Run r = { "failed", 0, format("the worker failed (status %d)", status) };
    FileStorage fs(result, FileStorage::READ);
    if (fs.isOpened())
        r = { (std::string)fs["status"], (double)fs["us"], (std::string)fs["why"] };
    fs.release();
    std::remove(rulesPath.c_str());
    std::remove(result.c_str());
    return r;
}

// every candidate timed on every case, REPEATS times in a rotating order
struct Pass
{
    std::vector<std::string> candidates;
    std::vector<Case> cases;
    std::map<std::string, std::vector<std::vector<double> > > times;    //!< [case][repeat]
    std::set<std::pair<std::string, size_t> > declined;
    std::map<std::string, std::string> failed;                          //!< wrong results or a crashed worker

    bool ran(const std::string& c, size_t ci) const
    {
        return !failed.count(c) && !declined.count(std::make_pair(c, ci)) && !times.at(c)[ci].empty();
    }
    double us(const std::string& c, size_t ci) const
    {
        return median(times.at(c)[ci]);
    }
};

Pass measure(const std::vector<std::string>& candidates, const std::vector<Case>& cases,
             const std::function<Run(const std::string&, const Case&)>& run)
{
    Pass p;
    p.candidates = candidates;
    p.cases = cases;
    for (const std::string& c : candidates)
        p.times[c].resize(cases.size());
    const size_t n = candidates.size();
    for (int rep = 0; rep < REPEATS; rep++)
        for (size_t ci = 0; ci < cases.size(); ci++)
            for (size_t j = 0; j < n; j++)
            {
                const std::string& c = candidates[(j + rep) % n];
                if (p.failed.count(c) || p.declined.count(std::make_pair(c, ci)))
                    continue;
                const Run r = run(c, cases[ci]);
                if (r.status == "ok")
                {
                    p.times[c][ci].push_back(r.us);
                    CV_LOG_DEBUG(NULL, format("    %-16s %-12s %10.1f us", c.c_str(), cases[ci].label.c_str(), r.us));
                }
                else if (r.status == "declined")
                {
                    p.declined.insert(std::make_pair(c, ci));
                    CV_LOG_INFO(NULL, "    " << c << "  " << cases[ci].label << "  declined: " << r.why);
                }
                else
                {
                    p.failed[c] = r.why;
                    CV_LOG_WARNING(NULL, "    " << c << "  " << cases[ci].label << "  failed, left out of the rules: " << r.why);
                }
            }
    return p;
}

std::vector<size_t> allCases(const Pass& p)
{
    std::vector<size_t> v(p.cases.size());
    std::iota(v.begin(), v.end(), (size_t)0);
    return v;
}

// the largest spread of the repeats, relative to their median, over @p cases
double noise(const Pass& p, const std::vector<size_t>& cases)
{
    double n = MIN_NOISE;
    for (const std::string& c : p.candidates)
        for (size_t ci : cases)
            if (p.ran(c, ci))
            {
                const std::vector<double>& t = p.times.at(c)[ci];
                n = std::max(n, (*std::max_element(t.begin(), t.end()) - *std::min_element(t.begin(), t.end())) / p.us(c, ci));
            }
    return n;
}

//! the first of @p order that ran on all of @p cases, or empty
std::string firstRan(const Pass& p, const std::vector<size_t>& cases, const std::vector<std::string>& order)
{
    for (const std::string& c : order)
    {
        bool all = true;
        for (size_t ci : cases)
            all = all && p.ran(c, ci);
        if (all)
            return c;
    }
    return std::string();
}

std::string fastest(const Pass& p, size_t ci)
{
    std::string best;
    for (const std::string& c : p.candidates)
        if (p.ran(c, ci) && (best.empty() || p.us(c, ci) < p.us(best, ci)))
            best = c;
    return best;
}

// faster by more than the noise overall, and slower by no more than the noise on any case
bool clearlyFaster(const Pass& p, const std::vector<size_t>& cases, const std::string& a, const std::string& b)
{
    const double n = noise(p, cases);
    double logRatio = 0;
    for (size_t ci : cases)
    {
        if (p.us(a, ci) > p.us(b, ci) * (1 + n))
            return false;
        logRatio += std::log(p.us(b, ci) / p.us(a, ci));
    }
    return std::exp(logRatio / cases.size()) - 1 > n;
}

// ranked by geometric mean over @p cases, except that the incumbent stays first unless another is clearly faster
std::vector<std::string> decide(const Pass& p, const std::vector<size_t>& cases, const std::vector<std::string>& incumbent)
{
    std::vector<std::string> ranked, rest;
    std::map<std::string, double> score;
    for (const std::string& c : p.candidates)
    {
        if (p.failed.count(c))
            continue;
        if (firstRan(p, cases, { c }).empty())
        {
            rest.push_back(c);
            continue;
        }
        double logSum = 0;
        for (size_t ci : cases)
            logSum += std::log(p.us(c, ci));
        score[c] = std::exp(logSum / cases.size());
        ranked.push_back(c);
    }
    std::stable_sort(ranked.begin(), ranked.end(), [&](const std::string& a, const std::string& b) { return score[a] < score[b]; });

    const std::string inc = firstRan(p, cases, incumbent);
    if (!inc.empty() && ranked[0] != inc && !clearlyFaster(p, cases, ranked[0], inc))
    {
        ranked.erase(std::find(ranked.begin(), ranked.end(), inc));
        ranked.insert(ranked.begin(), inc);
    }
    ranked.insert(ranked.end(), rest.begin(), rest.end());
    return ranked;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// rules from the measurements

// an op as the tuner sees it: candidates, the cases to time them on, and how a ranking maps to rules
struct OpSpec
{
    std::string op;
    std::vector<std::string> candidates;
    std::vector<Case> cases;
    std::function<rules::Choice(const std::vector<std::string>&)> toChoice;
    std::function<std::vector<std::string>(const rules::Choice&)> toOrder;
    std::function<Run(const std::string&, const Case&)> run;
};

std::vector<OpSpec> opSpecs(const std::string& self, const std::string& device)
{
    const Size sizes[] = { Size(1920, 1080), Size(3840, 2160) };
    const char* const sizeLabels[] = { "FHD", "4K" };

    // the LDS tile exists for 3x3 and 5x5 only; larger windows have nothing to choose from
    OpSpec box;
    box.op = "boxFilter";
    box.candidates = { "lds_tile", "sliding_column" };
    for (int cn : { 1, 3, 4 })
        for (int k : { 3, 5 })
            for (int s = 0; s < 2; s++)
                box.cases.push_back({ format("%s C%d k%d", sizeLabels[s], cn, k), sizes[s], cn, k });
    box.toChoice = [](const std::vector<std::string>& order) { rules::Choice c; c.kernels = order; return c; };
    box.toOrder = [](const rules::Choice& c) { return c.kernels; };
    box.run = [self, device](const std::string& kernel, const Case& c) {
        return runWorker(self, device, "boxFilter", kernel, Size(), c);
    };

    // the block shape reaches the kernel at launch, so these need no rebuild
    OpSpec nv;
    nv.op = "nv12ToBgr";
    nv.candidates = { "64x4", "32x8", "128x2", "256x1" };
    for (int cn : { 3, 4 })
        for (int s = 0; s < 2; s++)
            nv.cases.push_back({ format("%s C%d", sizeLabels[s], cn), sizes[s], cn, 0 });
    nv.toChoice = [](const std::vector<std::string>& order) {
        rules::Choice c;
        c.kernels = { "chroma_share" };
        parseSize(order[0], c.block);
        return c;
    };
    nv.toOrder = [](const rules::Choice& c) {
        std::vector<std::string> v;
        if (std::find(c.kernels.begin(), c.kernels.end(), "chroma_share") != c.kernels.end())
            v.push_back(sizeName(c.block.empty() ? Size(64, 4) : c.block));
        return v;
    };
    nv.run = [self, device](const std::string& block, const Case& c) {
        Size b;
        parseSize(block, b);
        return runWorker(self, device, "nv12ToBgr", "chroma_share", b, c);
    };
    return { box, nv };
}

bool sameChoice(const rules::Choice& a, const rules::Choice& b)
{
    return a.kernels == b.kernels && a.block == b.block;
}

// joins workload rules that choose the same: first over ksize, then over cn
void merge(std::vector<rules::Choice>& w)
{
    for (int pass = 0; pass < 2; pass++)
        for (size_t i = 0; i < w.size(); i++)
            for (size_t j = i + 1; j < w.size();)
            {
                std::vector<int>& same = pass == 0 ? w[i].cn : w[i].ksize;
                std::vector<int>& joins = pass == 0 ? w[i].ksize : w[i].cn;
                const std::vector<int>& otherSame = pass == 0 ? w[j].cn : w[j].ksize;
                const std::vector<int>& otherJoins = pass == 0 ? w[j].ksize : w[j].cn;
                if (sameChoice(w[i], w[j]) && same == otherSame)
                {
                    joins.insert(joins.end(), otherJoins.begin(), otherJoins.end());
                    w.erase(w.begin() + j);
                }
                else
                    j++;
            }
}

std::string evidence(const Pass& p)
{
    std::string e;
    for (size_t ci = 0; ci < p.cases.size(); ci++)
    {
        e += (ci ? "; " : "") + p.cases[ci].label + ":";
        for (const std::string& c : p.candidates)
            if (p.ran(c, ci))
                e += format(" %s %.1f", c.c_str(), p.us(c, ci));
    }
    return e + format(" (us); spread %.1f%%", noise(p, allCases(p)) * 100);
}

// the ranking over every case, plus a workload rule for each (cn, ksize) that a different candidate wins
rules::OpRules decideOp(const OpSpec& s, const Pass& p, const rules::OpRules* shipped)
{
    rules::OpRules r;
    const std::vector<std::string> general = decide(p, allCases(p), shipped ? s.toOrder(shipped->general)
                                                                            : std::vector<std::string>());
    if (general.empty())
        return r;
    r.general = s.toChoice(general);

    std::map<std::pair<int, int>, std::vector<size_t> > classes;
    for (size_t ci = 0; ci < p.cases.size(); ci++)
        classes[std::make_pair(p.cases[ci].cn, p.cases[ci].k)].push_back(ci);
    for (const auto& cls : classes)
    {
        const rules::Workload w = { cls.first.first, cls.first.second };
        const rules::Choice* rule = shipped && &shipped->choose(w) != &shipped->general ? &shipped->choose(w) : 0;
        const std::vector<std::string> local = decide(p, cls.second, rule ? s.toOrder(*rule) : general);
        const std::string pick = firstRan(p, cls.second, local);
        if (pick.empty() || pick == firstRan(p, cls.second, general))
            continue;
        rules::Choice c = s.toChoice(local);
        c.cn.push_back(w.cn);
        if (w.ksize)
            c.ksize.push_back(w.ksize);
        r.workloads.push_back(c);
    }
    merge(r.workloads);
    r.evidence = evidence(p);
    return r;
}

std::string describe(const rules::Choice& c)
{
    std::string s;
    if (!c.cn.empty())
        s += "cn " + joined(c.cn) + " ";
    if (!c.ksize.empty())
        s += "ksize " + joined(c.ksize) + " ";
    s += "[" + joined(c.kernels) + "]";
    return c.block.empty() ? s : s + " block " + sizeName(c.block);
}

std::string describe(const rules::OpRules& r)
{
    std::string s = describe(r.general);
    for (const rules::Choice& w : r.workloads)
        s += "; " + describe(w);
    return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// report

// what the plugin runs for case @p ci under @p op: the first candidate of its choice that ran; empty for the CPU
std::string selected(const OpSpec& s, const Pass& p, size_t ci, const rules::OpRules* op)
{
    if (!op)
        return std::string();
    for (const std::string& c : s.toOrder(op->choose({ p.cases[ci].cn, p.cases[ci].k })))
        if (p.ran(c, ci))
            return c;
    return std::string();
}

// stability of tuning across passes @p a and @p b, and regret on @p b of the installed rules and of those tuned on @p a
void report(const OpSpec& s, const Pass& a, const Pass& b, const rules::OpRules* installed, const rules::OpRules* shipped)
{
    const rules::OpRules tunedA = decideOp(s, a, shipped), tunedB = decideOp(s, b, shipped);
    CV_LOG_INFO(NULL, "  tuned on pass 1: " << describe(tunedA));
    if (describe(tunedA) == describe(tunedB))
    {
        CV_LOG_INFO(NULL, "  tuned on pass 2: the same, stable");
    }
    else
    {
        CV_LOG_WARNING(NULL, "  tuned on pass 2: " << describe(tunedB) << ", NOT stable");
    }
    int flips = 0;
    for (size_t ci = 0; ci < a.cases.size(); ci++)
        flips += fastest(a, ci) != fastest(b, ci);
    CV_LOG_INFO(NULL, format("  fastest candidate differs between the passes on %d of %d cases", flips, (int)a.cases.size()));

    const double n = noise(b, allCases(b));
    CV_LOG_INFO(NULL, format("  regret on pass 2 against its fastest candidate (noise %.1f%%): installed | tuned on pass 1", n * 100));
    const rules::OpRules* sets[] = { installed, tunedA.general.kernels.empty() ? 0 : &tunedA };
    const char* const names[] = { "installed", "tuned on pass 1" };
    double sum[2] = { 0, 0 }, worst[2] = { 0, 0 };
    std::string worstCase[2];
    int timed[2] = { 0, 0 }, cpu[2] = { 0, 0 }, beyond[2] = { 0, 0 };
    for (size_t ci = 0; ci < b.cases.size(); ci++)
    {
        const std::string best = fastest(b, ci);
        if (best.empty())
        {
            CV_LOG_INFO(NULL, "    " << b.cases[ci].label << ": no candidate ran");
            continue;
        }
        std::string line = format("    %-12s fastest %-14s %8.1f us", b.cases[ci].label.c_str(), best.c_str(), b.us(best, ci));
        for (int i = 0; i < 2; i++)
        {
            const std::string sel = selected(s, b, ci, sets[i]);
            if (sel.empty())
            {
                line += " | CPU";
                cpu[i]++;
                continue;
            }
            const double x = b.us(sel, ci) / b.us(best, ci) - 1;
            line += format(" | %s %+.1f%%", sel.c_str(), x * 100);
            sum[i] += x;
            timed[i]++;
            beyond[i] += x > n;
            if (x > worst[i])
            {
                worst[i] = x;
                worstCase[i] = b.cases[ci].label;
            }
        }
        CV_LOG_INFO(NULL, line);
    }
    for (int i = 0; i < 2; i++)
        CV_LOG_INFO(NULL, format("  %s rules: mean regret %.1f%%, worst %.1f%%%s, %d of %d cases beyond the noise%s",
                                 names[i], timed[i] ? sum[i] / timed[i] * 100 : 0., worst[i] * 100,
                                 worstCase[i].empty() ? "" : (" (" + worstCase[i] + ")").c_str(), beyond[i], timed[i],
                                 cpu[i] ? format(", %d on the CPU", cpu[i]).c_str() : ""));
}

////////////////////////////////////////////////////////////////////////////////////////////////
// driver

void writeChoice(FileStorage& fs, const rules::Choice& c)
{
    if (!c.cn.empty())
    {
        fs << "cn" << "[:";
        for (int v : c.cn)
            fs << v;
        fs << "]";
    }
    if (!c.ksize.empty())
    {
        fs << "ksize" << "[:";
        for (int v : c.ksize)
            fs << v;
        fs << "]";
    }
    fs << "kernels" << "[:";
    for (const std::string& k : c.kernels)
        fs << k;
    fs << "]";
    if (!c.block.empty())
        fs << "block" << "[:" << c.block.width << c.block.height << "]";
}

void writeTuned(const std::string& path, const rules::RuleFile& sections)
{
    const size_t slash = path.rfind('/');
    if (slash != std::string::npos)
        utils::fs::createDirectories(path.substr(0, slash));
    const std::string tmp = path + ".tmp";
    {
        FileStorage fs(tmp, FileStorage::WRITE | FileStorage::FORMAT_YAML);
        if (!fs.isOpened())
            CV_Error_(Error::StsError, ("cannot write %s", tmp.c_str()));
        fs.writeComment(" Written by opencv_hip_tune; used ahead of imgproc.yaml for the GPU, HIP runtime and plugin");
        fs.writeComment(" build each section was measured with. opencv_hip_tune --reset deletes it.");
        fs << "schema" << rules::SCHEMA << "archs" << "{";
        for (rules::RuleFile::const_iterator it = sections.begin(); it != sections.end(); ++it)
        {
            const rules::ArchRules& s = it->second;
            fs << it->first << "{";
            fs << "measured_on" << "{" << "device" << s.device << "hip_runtime" << s.runtime
               << "plugin" << s.plugin << "date" << s.date << "}";
            for (std::map<std::string, rules::OpRules>::const_iterator op = s.ops.begin(); op != s.ops.end(); ++op)
            {
                fs << op->first << "{";
                writeChoice(fs, op->second.general);
                if (!op->second.workloads.empty())
                {
                    fs << "workloads" << "[";
                    for (const rules::Choice& w : op->second.workloads)
                    {
                        fs << "{:";
                        writeChoice(fs, w);
                        fs << "}";
                    }
                    fs << "]";
                }
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

void hipCheck(hipError_t e, const char* what)
{
    if (e != hipSuccess)
        CV_Error_(Error::GpuApiCallError, ("%s failed: %s", what, hipGetErrorString(e)));
}

std::vector<accel::Device> hipDevices(const std::string& name)
{
    std::vector<accel::Device> devs;
    for (const accel::Device& d : accel::Device::enumerate())
        if (d.backend() == "hip" && (name.empty() || d.name() == name))
            devs.push_back(d);
    if (devs.empty())
        CV_Error_(Error::StsError, ("no HIP GPU %s", name.c_str()));
    return devs;
}

const rules::OpRules* findOp(const rules::ArchRules& a, const std::string& op)
{
    std::map<std::string, rules::OpRules>::const_iterator it = a.ops.find(op);
    return it == a.ops.end() ? 0 : &it->second;
}

int check(const Options& o)
{
    std::vector<std::string> warnings;
    if (!o.checkFile.empty())
    {
        const rules::RuleFile r = rules::read(o.checkFile, warnings);
        for (rules::RuleFile::const_iterator it = r.begin(); it != r.end(); ++it)
            CV_LOG_INFO(NULL, o.checkFile << ", " << it->first << ": " << it->second.ops.size() << " operation(s)");
    }
    else
    {
        const std::string dir = tuningDir();
        const rules::RuleFile shipped = rules::read(dir + "/imgproc.yaml", warnings);
        const std::string tunedPath = dir + "/imgproc.tuned.yaml";
        const rules::RuleFile tuned = utils::fs::exists(tunedPath) ? rules::read(tunedPath, warnings) : rules::RuleFile();
        int runtime = 0;
        hipCheck(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
        const std::string plugin = rules::fingerprint(pluginPath());
        for (const accel::Device& dev : hipDevices(o.device))
        {
            hipDeviceProp_t p;
            hipCheck(hipGetDeviceProperties(&p, dev.index()), "hipGetDeviceProperties");
            const std::string arch = std::string(p.gcnArchName).substr(0, std::string(p.gcnArchName).find(':'));
            bool found = false;
            const rules::ArchRules a = rules::effective(shipped, tuned, arch, p.name, runtime, plugin, warnings, found);
            if (!found)
                warnings.push_back(dev.name() + " (" + arch + "): no rules, every operation runs on the CPU");
            for (std::map<std::string, rules::OpRules>::const_iterator op = a.ops.begin(); op != a.ops.end(); ++op)
                CV_LOG_INFO(NULL, dev.name() << " " << op->first << " [rules: " << a.section << "]: " << describe(op->second));
        }
    }
    logWarnings(warnings);
    CV_LOG_INFO(NULL, (warnings.empty() ? std::string("no problems") : format("%d problem(s)", (int)warnings.size())));
    return warnings.empty() ? 0 : 1;
}

void tune(const Options& o)
{
    const std::string dir = tuningDir();
    const std::string out = o.output.empty() ? dir + "/imgproc.tuned.yaml" : o.output;
    if (o.reset)
    {
        if (utils::fs::exists(out) && std::remove(out.c_str()) != 0)
            CV_Error_(Error::StsError, ("cannot delete %s", out.c_str()));
        CV_LOG_INFO(NULL, "deleted " << out << ": the shipped rules apply from the next program start");
        return;
    }

    const std::vector<accel::Device> devs = hipDevices(o.device);
    const std::string self = selfPath(), plugin = rules::fingerprint(pluginPath());
    std::vector<std::string> warnings;
    const rules::RuleFile shipped = rules::read(dir + "/imgproc.yaml", warnings);
    rules::RuleFile tuned = utils::fs::exists(out) ? rules::read(out, warnings) : rules::RuleFile();
    logWarnings(warnings);
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
        std::vector<std::string> w;
        bool found = false;
        const rules::ArchRules shippedArch = rules::effective(shipped, rules::RuleFile(), arch, "", 0, "", w, found);
        const rules::ArchRules installed = rules::effective(shipped, tuned, arch, p.name, runtime, plugin, w, found);
        logWarnings(w);
        CV_LOG_INFO(NULL, dev.name() << "  " << p.name << "  " << arch << "  (installed rules: "
                    << (found ? installed.section : std::string("none")) << ")");

        rules::ArchRules s;
        s.section = arch;
        s.device = p.name;
        s.runtime = runtime;
        s.plugin = plugin;
        s.date = date;
        for (const OpSpec& spec : opSpecs(self, dev.name()))
        {
            CV_LOG_INFO(NULL, "  " << spec.op << ": " << spec.cases.size() << " cases, candidates " << joined(spec.candidates));
            const Pass a = measure(spec.candidates, spec.cases, spec.run);
            if (o.report)
            {
                report(spec, a, measure(spec.candidates, spec.cases, spec.run), findOp(installed, spec.op),
                       findOp(shippedArch, spec.op));
                continue;
            }
            const rules::OpRules r = decideOp(spec, a, findOp(shippedArch, spec.op));
            if (r.general.kernels.empty())
            {
                CV_LOG_WARNING(NULL, "  every candidate failed: " << spec.op << " keeps the shipped rules");
                continue;
            }
            CV_LOG_INFO(NULL, "  -> " << describe(r));
            CV_LOG_INFO(NULL, "     " << r.evidence);
            s.ops[spec.op] = r;
        }
        if (o.report)
            continue;
        if (tuned.count(arch) && tuned[arch].device != s.device)
            CV_LOG_INFO(NULL, "  (replaces the section measured on " << tuned[arch].device << ")");
        tuned[arch] = s;
    }

    if (o.report || o.dryRun)
    {
        CV_LOG_INFO(NULL, "nothing written");
        return;
    }
    writeTuned(out, tuned);
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
        else if (a == "--report") o.report = true;
        else if (a == "--check")
        {
            o.check = true;
            if (hasValue && argv[i + 1][0] != '-')
                o.checkFile = argv[++i];
        }
        else if (a == "--device" && hasValue) o.device = argv[++i];
        else if (a == "--output" && hasValue) o.output = argv[++i];
        else if (a == "--op" && hasValue) o.op = argv[++i];
        else if (a == "--result" && hasValue) o.result = argv[++i];
        else if (a == "--cn" && hasValue) o.cn = std::atoi(argv[++i]);
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
    else if (o.check)
        return check(o);
    else
        tune(o);
    return 0;
}
