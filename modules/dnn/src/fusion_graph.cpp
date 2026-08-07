// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "fusion_graph.hpp"
#include <cmath>

namespace cv { namespace dnn {

bool isCommutativeFusionOp(FusionOp op){
    switch(op){
        case FusionOp::ADD:
        case FusionOp::MUL:
            return true;
        default:
            return false;
    }
}

static int nOperands(FusionOp op)
{
    switch (op) {
        case FusionOp::INPUT:
        case FusionOp::CONST:
        case FusionOp::PER_CHANNEL_CONST:
            return 0;
        case FusionOp::ADD: case FusionOp::SUB: case FusionOp::MUL:
        case FusionOp::MAX: case FusionOp::MIN:
            return 2;
        case FusionOp::ERF: case FusionOp::TANH: case FusionOp::EXP:
        case FusionOp::SQRT: case FusionOp::CLAMP: case FusionOp::RECIP:
            return 1;
    }
    CV_Error(cv::Error::StsBadArg, "unhandled FusionOp in nOperands");
}

int FusionGraphBuilder::push(FusionOp op, std::vector<int> inputs,
                    float scalar, float scalar2, int constBufferId)
{
    CV_Assert((g.size() == 0) == (op == FusionOp::INPUT));
    CV_Assert((int)inputs.size() == nOperands(op));
    CV_Assert(g.size() < (size_t)FUSION_MAX_NODES);
    for (int i : inputs)
        CV_Assert(i >= 0 && i < (int)g.size());

    if (isCommutativeFusionOp(op) && inputs.size() == 2 && inputs[0] > inputs[1])
        std::swap(inputs[0], inputs[1]);

    if (op != FusionOp::CONST && op != FusionOp::CLAMP) scalar  = 0.f;
    if (op != FusionOp::CLAMP)                      scalar2 = 0.f;
    if (op != FusionOp::PER_CHANNEL_CONST)          constBufferId = -1;

    FusionNode candidate{op, inputs, scalar, scalar2, constBufferId};

    auto it = seen.find(candidate);
    if (it != seen.end())
        return it->second;

    int idx = g.append(candidate);
    seen.emplace(std::move(candidate), idx);
    return idx;
}

static int sideNode(FusionGraphBuilder& b, const FusionOperand& oper)
{
    if (oper.perChannel)
        return b.push(FusionOp::PER_CHANNEL_CONST, {}, 0.f, 0.f, oper.bufId);
    return b.push(FusionOp::CONST, {}, oper.scalar);
}

enum { FUSION_APPEND_HEADROOM = 8 };

int appendFusionNode(FusionGraphBuilder& b, int cur, const std::string& opType, const FusionOperand& oper)
{
    if (cur < 0)
        return -1;
    if (b.g.size() + FUSION_APPEND_HEADROOM > (size_t)FUSION_MAX_NODES)
        return -1;

    if (opType == "Relu")
        return b.push(FusionOp::MAX, {cur, b.push(FusionOp::CONST, {}, 0.f)});
    if (opType == "Sqrt") return b.push(FusionOp::SQRT, {cur});
    if (opType == "Exp")  return b.push(FusionOp::EXP,  {cur});
    if (opType == "Tanh") return b.push(FusionOp::TANH, {cur});
    if (opType == "Erf")  return b.push(FusionOp::ERF,  {cur});

    if (opType == "Sigmoid") {
        int one = b.push(FusionOp::CONST, {}, 1.f);
        int neg = b.push(FusionOp::MUL, {cur, b.push(FusionOp::CONST, {}, -1.f)});
        return b.push(FusionOp::RECIP, {b.push(FusionOp::ADD, {one, b.push(FusionOp::EXP, {neg})})});
    }

    if (opType == "Gelu") {
        int half = b.push(FusionOp::CONST, {}, 0.5f);
        int one  = b.push(FusionOp::CONST, {}, 1.f);
        int inv  = b.push(FusionOp::CONST, {}, 0.70710678118654752f);
        int e    = b.push(FusionOp::ERF, {b.push(FusionOp::MUL, {cur, inv})});
        int s    = b.push(FusionOp::ADD, {one, e});
        return b.push(FusionOp::MUL, {b.push(FusionOp::MUL, {half, cur}), s});
    }

    if (opType == "Clip")
        return b.push(FusionOp::CLAMP, {cur}, oper.scalar, oper.scalar2);

    if (!oper.hasSide)
        return -1;

    if (opType == "Add") return b.push(FusionOp::ADD, {cur, sideNode(b, oper)});
    if (opType == "Sub") return b.push(FusionOp::SUB, {cur, sideNode(b, oper)});
    if (opType == "Mul") return b.push(FusionOp::MUL, {cur, sideNode(b, oper)});
    if (opType == "Max") return b.push(FusionOp::MAX, {cur, sideNode(b, oper)});
    if (opType == "Min") return b.push(FusionOp::MIN, {cur, sideNode(b, oper)});

    return -1;
}

float evalFusionGraph(const FusionGraph& g, float x,
                    const std::vector<const float*>& constBufs, int channelIdx)
{
    const std::vector<FusionNode>& nodes = g.nodes();
    CV_Assert(g.outputNode >= 0 && g.outputNode < (int)nodes.size());
    CV_Assert(nodes.size() <= (size_t)FUSION_MAX_NODES);

    float v[FUSION_MAX_NODES];

    for (size_t i = 0; i < nodes.size(); i++) {
        const FusionNode& n = nodes[i];
        const float a = n.inputs.size() > 0 ? v[n.inputs[0]] : 0.f;
        const float b = n.inputs.size() > 1 ? v[n.inputs[1]] : 0.f;

        switch (n.op) {
        case FusionOp::INPUT: v[i] = x; break;
        case FusionOp::CONST: v[i] = n.scalar; break;
        case FusionOp::PER_CHANNEL_CONST:
            CV_DbgAssert(n.constBufferId >= 0 && n.constBufferId < (int)constBufs.size());
            v[i] = constBufs[n.constBufferId][channelIdx];
            break;
        case FusionOp::ADD:  v[i] = a + b; break;
        case FusionOp::SUB:  v[i] = a - b; break;
        case FusionOp::MUL:  v[i] = a * b; break;
        case FusionOp::MAX:  v[i] = a < b ? b : a; break;
        case FusionOp::MIN:  v[i] = b < a ? b : a; break;
        case FusionOp::ERF:  v[i] = std::erf(a); break;
        case FusionOp::TANH: v[i] = std::tanh(a); break;
        case FusionOp::EXP:  v[i] = std::exp(a); break;
        case FusionOp::SQRT:  v[i] = std::sqrt(a); break;
        case FusionOp::RECIP: v[i] = 1.f / a; break;
        case FusionOp::CLAMP:
            v[i] = a < n.scalar ? n.scalar : (a > n.scalar2 ? n.scalar2 : a);
            break;
        }
    }
    return v[g.outputNode];
}

}// namespace dnn
}// namespace cv
