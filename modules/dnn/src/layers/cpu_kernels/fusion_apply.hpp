// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__
#define __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__

#include "opencv2/core.hpp"
#include "opencv2/dnn/all_layers.hpp"
#include "../../fusion_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct FusionApply
{
    Ptr<FusionGraph> expr;
    ActivationFunc fn = nullptr;
    std::vector<float> params;
    std::vector<const float*> bufPtrs;
};

inline const std::vector<std::pair<int, Ptr<FusionGraph> > >& fusionActivationRefs()
{
    static const std::vector<std::pair<int, Ptr<FusionGraph> > > refs = []
    {
        std::vector<std::pair<int, Ptr<FusionGraph> > > v;
        FusionRecipe r;
        r = FusionRecipe(); sigmoidRecipe(r); v.push_back(std::make_pair(ACTIV_SIGMOID, graphFromRecipe(r)));
        r = FusionRecipe(); geluRecipe(r);    v.push_back(std::make_pair(ACTIV_GELU,    graphFromRecipe(r)));
        return v;
    }();
    return refs;
}

inline bool matchFusionActivation(const FusionGraph& g, int& activType,
                                  std::vector<float>& params)
{
    const std::vector<FusionNode>& nd = g.nodes();
    if (nd.empty() || nd[0].op != FusionEltwiseOp::INPUT)
        return false;
    if (g.outputNode != (int)nd.size() - 1)
        return false;

    if (nd.size() == 2 && nd[1].inputs.size() == 1 && nd[1].inputs[0] == 0) {
        switch (nd[1].op) {
        case FusionEltwiseOp::CLAMP:
            activType = ACTIV_CLIP;
            params.assign(2, 0.f);
            params[0] = nd[1].scalar;
            params[1] = nd[1].scalar2;
            return true;
        case FusionEltwiseOp::TANH:
            activType = ACTIV_TANH;
            params.clear();
            return true;
        case FusionEltwiseOp::ERF:
            activType = ACTIV_ERF;
            params.clear();
            return true;
        case FusionEltwiseOp::EXP:
            activType = ACTIV_EXP;
            params.assign(2, 0.f);
            params[0] = 1.f;
            return true;
        default:
            break;
        }
    }

    if (nd.size() == 3 &&
        nd[1].op == FusionEltwiseOp::CONST && bitsOf(nd[1].scalar) == bitsOf(0.f) &&
        nd[2].op == FusionEltwiseOp::MAX && nd[2].inputs.size() == 2 &&
        nd[2].inputs[0] == 0 && nd[2].inputs[1] == 1) {
        activType = ACTIV_RELU;
        params.assign(1, 0.f);
        return true;
    }

    const std::vector<std::pair<int, Ptr<FusionGraph> > >& refs = fusionActivationRefs();
    for (size_t i = 0; i < refs.size(); i++) {
        if (refs[i].second && sameFusionGraph(g, *refs[i].second)) {
            activType = refs[i].first;
            params.clear();
            return true;
        }
    }
    return false;
}

inline bool prepareFusionApply(const Ptr<FusionGraph>& expr, FusionApply& out)
{
    if (!expr || expr->size() == 0)
        return false;
    if (expr->outputNode != (int)expr->size() - 1)
        return false;
    if (expr->size() > (size_t)FUSION_MAX_NODES)
        return false;

    out = FusionApply();
    out.expr = expr;

    int activType = ACTIV_NONE;
    if (matchFusionActivation(*expr, activType, out.params)) {
        out.fn = getActivationFunc(activType);
        if (out.fn)
            return true;
    }

    out.fn = nullptr;
    out.params.clear();
    out.bufPtrs.resize(expr->constBufs.size());
    for (size_t i = 0; i < expr->constBufs.size(); i++) {
        const Mat& m = expr->constBufs[i];
        if (m.empty() || m.type() != CV_32F || !m.isContinuous())
            return false;
        out.bufPtrs[i] = m.ptr<float>();
    }
    for (const FusionNode& n : expr->nodes()) {
        if (n.op == FusionEltwiseOp::PER_CHANNEL_CONST &&
            (n.constBufferId < 0 || n.constBufferId >= (int)out.bufPtrs.size()))
            return false;
    }
    return true;
}

inline void applyFusion(const FusionApply& a, Mat& Y)
{
    if (!a.fn && !a.expr)
        return;
    CV_Assert(Y.type() == CV_32F && Y.isContinuous());

    float* p = Y.ptr<float>();
    const size_t n = Y.total();
    if (n == 0)
        return;

    size_t BLOCK = 1 << 16;
    const size_t nt = (size_t)std::max(1, getNumThreads());
    if (nt > 1 && n < BLOCK * nt)
        BLOCK = std::max<size_t>(1 << 12, (n + nt - 1) / nt);
    const int nblocks = (int)((n + BLOCK - 1) / BLOCK);

    if (a.fn) {
        const float* pr = a.params.empty() ? nullptr : a.params.data();
        parallel_for_(Range(0, nblocks), [&](const Range& r) {
            for (int b = r.start; b < r.end; b++) {
                const size_t st = (size_t)b * BLOCK, en = std::min(st + BLOCK, n);
                a.fn(p + st, p + st, en - st, pr);
            }
        });
        return;
    }

    const int nch = Y.dims > 0 ? std::max(Y.size[Y.dims - 1], 1) : 1;
    for (size_t i = 0; i < a.bufPtrs.size(); i++)
        CV_DbgAssert((int)a.expr->constBufs[i].total() == nch);

    const FusionGraph& g = *a.expr;
    const std::vector<const float*>& bufs = a.bufPtrs;
    parallel_for_(Range(0, nblocks), [&](const Range& r) {
        for (int b = r.start; b < r.end; b++) {
            const size_t st = (size_t)b * BLOCK, en = std::min(st + BLOCK, n);
            int c = (int)(st % (size_t)nch);
            for (size_t k = st; k < en; k++) {
                p[k] = evalFusionGraph(g, p[k], bufs, c);
                if (++c == nch) c = 0;
            }
        }
    });
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
