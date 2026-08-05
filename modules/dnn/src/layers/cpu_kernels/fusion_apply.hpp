// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__
#define __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__

#include "opencv2/core.hpp"
#include "opencv2/dnn/all_layers.hpp"
#include "../../graph_fusion_utils.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct FusionStep
{
    ActivationFunc fn = nullptr;
    std::vector<float> params;
    Mat bias;
    float scalar = 0.f;
    bool scalarAdd = false;
    bool interp = false;
    FusionGraph g;
    std::vector<Mat> bufs;
};

inline bool fusionLowerActiv(const Ptr<LayerInfo>& l, FusionStep& s)
{
    const ActivationLayer* a = dynamic_cast<const ActivationLayer*>(l.get());
    if (!a)
        return false;
    s.fn = a->getActivationFunc(CV_32F, s.params);
    return s.fn != nullptr;
}

inline bool fusionLowerClip(const AgnosticChain& ch, size_t i, FusionStep& s)
{
    if (effectiveOpType(ch.absorbed[i]) != "Clip" || (int)i >= ch.fgSteps)
        return false;
    const FusionOperand& o = ch.stepOperands[i];
    s.params.assign(2, 0.f);
    s.params[0] = o.scalar;
    s.params[1] = o.scalar2;
    s.fn = getActivationFunc(ACTIV_CLIP);
    return s.fn != nullptr;
}

inline bool fusionLowerBiasAdd(const AgnosticChain& ch, size_t i, FusionStep& s)
{
    if (effectiveOpType(ch.absorbed[i]) != "Add")
        return false;
    const FusionOperand& o = ch.stepOperands[i];
    if (!o.hasSide)
        return false;
    if (o.bufId < 0) {
        s.scalarAdd = true;
        s.scalar = o.scalar;
        return true;
    }
    if (o.bufId >= (int)ch.constBufs.size())
        return false;
    const Mat& c = ch.constBufs[o.bufId];
    if (c.empty() || c.type() != CV_32F || !c.isContinuous())
        return false;
    if (c.dims > 1 && c.size[c.dims - 1] == 1)
        return false;
    s.bias = c;
    return true;
}

inline bool fusionLower(const AgnosticChain& ch, std::vector<FusionStep>& out)
{
    if (ch.absorbed.empty() || ch.stepOperands.size() != ch.absorbed.size())
        return false;

    std::vector<FusionStep> steps;
    size_t i = 0;
    for (; i < ch.absorbed.size(); i++) {
        FusionStep s;
        if (!fusionLowerActiv(ch.absorbed[i], s) && !fusionLowerClip(ch, i, s) && !fusionLowerBiasAdd(ch, i, s))
            break;
        steps.push_back(s);
    }
    if (i == ch.absorbed.size()) {
        out = steps;
        return true;
    }

    if ((int)i >= ch.fgSteps || ch.fg.empty() || ch.fg.size() > (size_t)FUSION_MAX_NODES ||
        !fusionInterpEnabled())
        return false;

    steps.clear();
    FusionStep s;
    s.interp = true;
    s.g = ch.fg;
    s.bufs = ch.constBufs;
    for (const Mat& m : s.bufs) {
        if (m.empty() || m.type() != CV_32F || !m.isContinuous())
            return false;
        if (m.dims > 1 && m.size[m.dims - 1] == 1)
            return false;
    }
    steps.push_back(s);

    for (size_t k = (size_t)ch.fgSteps; k < ch.absorbed.size(); k++) {
        FusionStep t;
        if (!fusionLowerActiv(ch.absorbed[k], t) && !fusionLowerClip(ch, k, t) && !fusionLowerBiasAdd(ch, k, t))
            return false;
        steps.push_back(t);
    }
    out = steps;
    return true;
}

inline void fusionApply(const std::vector<FusionStep>& steps, Mat& Y)
{
    if (steps.empty())
        return;
    CV_Assert(Y.type() == CV_32F && Y.isContinuous());
    float* p = Y.ptr<float>();
    const size_t n = Y.total();

    const size_t BLOCK = 1 << 16;
    const int nblocks = (int)((n + BLOCK - 1) / BLOCK);

    const int nch = Y.dims > 0 ? Y.size[Y.dims - 1] : 1;

    for (const FusionStep& s : steps) {
        if (s.interp) {
            std::vector<const float*> bp(s.bufs.size());
            for (size_t i = 0; i < s.bufs.size(); i++) {
                CV_Assert((int)s.bufs[i].total() == nch);
                bp[i] = s.bufs[i].ptr<float>();
            }
            parallel_for_(Range(0, nblocks), [&](const Range& r) {
                for (int b = r.start; b < r.end; b++) {
                    size_t st = (size_t)b * BLOCK, en = std::min(st + BLOCK, n);
                    for (size_t k = st; k < en; k++)
                        p[k] = evalFusionGraph(s.g, p[k], bp, nch > 0 ? (int)(k % (size_t)nch) : 0);
                }
            });
        } else if (s.fn) {
            const float* prm = s.params.empty() ? nullptr : s.params.data();
            parallel_for_(Range(0, nblocks), [&](const Range& r) {
                for (int b = r.start; b < r.end; b++) {
                    size_t st = (size_t)b * BLOCK;
                    s.fn(p + st, p + st, std::min(BLOCK, n - st), prm);
                }
            });
        } else if (s.scalarAdd) {
            const float v = s.scalar;
            parallel_for_(Range(0, nblocks), [&](const Range& r) {
                for (int b = r.start; b < r.end; b++) {
                    size_t st = (size_t)b * BLOCK, en = std::min(st + BLOCK, n);
                    for (size_t k = st; k < en; k++)
                        p[k] += v;
                }
            });
        } else {
            const size_t N = s.bias.total();
            CV_Assert(N > 0 && n % N == 0);
            const float* bp = s.bias.ptr<float>();
            const int nrows = (int)(n / N);
            parallel_for_(Range(0, nrows), [&](const Range& r) {
                for (int i = r.start; i < r.end; i++) {
                    float* row = p + (size_t)i * N;
                    for (size_t j = 0; j < N; j++)
                        row[j] += bp[j];
                }
            });
        }
    }
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
