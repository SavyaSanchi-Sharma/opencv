// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "epilogue.hpp"
#include <cmath>

namespace cv { namespace dnn {

bool isCommutativeOP(EpOP op){
    switch(op){
        case EpOP::ADD:
        case EpOP::MUL:
            return true;
        default:
            return false;
    }
}

static int nOperands(EpOP op)
{
    switch (op) {
        case EpOP::INPUT:
        case EpOP::CONST:
        case EpOP::PER_CHANNEL_CONST:
            return 0;
        case EpOP::ADD: case EpOP::SUB: case EpOP::MUL:
        case EpOP::MAX: case EpOP::MIN:
            return 2;
        case EpOP::ERF: case EpOP::TANH: case EpOP::EXP:
        case EpOP::SQRT: case EpOP::CLAMP: case EpOP::RECIP:
            return 1;
    }
    CV_Error(cv::Error::StsBadArg, "unhandled EpOP in nOperands");
}

int EpBuilder::push(EpOP op, std::vector<int> inputs,
                    float scalar, float scalar2, int constBufferId)
{
    CV_Assert((g.size() == 0) == (op == EpOP::INPUT));
    CV_Assert((int)inputs.size() == nOperands(op));
    CV_Assert(g.size() < (size_t)EP_MAX_NODES);
    for (int i : inputs)
        CV_Assert(i >= 0 && i < (int)g.size());

    if (isCommutativeOP(op) && inputs.size() == 2 && inputs[0] > inputs[1])
        std::swap(inputs[0], inputs[1]);

    if (op != EpOP::CONST && op != EpOP::CLAMP) scalar  = 0.f;
    if (op != EpOP::CLAMP)                      scalar2 = 0.f;
    if (op != EpOP::PER_CHANNEL_CONST)          constBufferId = -1;

    EpNode candidate{op, inputs, scalar, scalar2, constBufferId};

    auto it = seen.find(candidate);
    if (it != seen.end())
        return it->second;

    int idx = g.append(candidate);
    seen.emplace(std::move(candidate), idx);
    return idx;
}

static int sideNode(EpBuilder& b, const EpOperand& oper)
{
    if (oper.perChannel)
        return b.push(EpOP::PER_CHANNEL_CONST, {}, 0.f, 0.f, oper.bufId);
    return b.push(EpOP::CONST, {}, oper.scalar);
}

enum { EP_APPEND_HEADROOM = 8 };

int appendNode(EpBuilder& b, int cur, const std::string& opType, const EpOperand& oper)
{
    if (cur < 0)
        return -1;
    if (b.g.size() + EP_APPEND_HEADROOM > (size_t)EP_MAX_NODES)
        return -1;

    if (opType == "Relu")
        return b.push(EpOP::MAX, {cur, b.push(EpOP::CONST, {}, 0.f)});
    if (opType == "Sqrt") return b.push(EpOP::SQRT, {cur});
    if (opType == "Exp")  return b.push(EpOP::EXP,  {cur});
    if (opType == "Tanh") return b.push(EpOP::TANH, {cur});
    if (opType == "Erf")  return b.push(EpOP::ERF,  {cur});

    if (opType == "Sigmoid") {
        int one = b.push(EpOP::CONST, {}, 1.f);
        int neg = b.push(EpOP::MUL, {cur, b.push(EpOP::CONST, {}, -1.f)});
        return b.push(EpOP::RECIP, {b.push(EpOP::ADD, {one, b.push(EpOP::EXP, {neg})})});
    }

    if (opType == "Gelu") {
        int half = b.push(EpOP::CONST, {}, 0.5f);
        int one  = b.push(EpOP::CONST, {}, 1.f);
        int inv  = b.push(EpOP::CONST, {}, 0.70710678118654752f);
        int e    = b.push(EpOP::ERF, {b.push(EpOP::MUL, {cur, inv})});
        int s    = b.push(EpOP::ADD, {one, e});
        return b.push(EpOP::MUL, {b.push(EpOP::MUL, {half, cur}), s});
    }

    if (opType == "Clip")
        return b.push(EpOP::CLAMP, {cur}, oper.scalar, oper.scalar2);

    if (!oper.hasSide)
        return -1;

    if (opType == "Add") return b.push(EpOP::ADD, {cur, sideNode(b, oper)});
    if (opType == "Sub") return b.push(EpOP::SUB, {cur, sideNode(b, oper)});
    if (opType == "Mul") return b.push(EpOP::MUL, {cur, sideNode(b, oper)});
    if (opType == "Max") return b.push(EpOP::MAX, {cur, sideNode(b, oper)});
    if (opType == "Min") return b.push(EpOP::MIN, {cur, sideNode(b, oper)});

    return -1;
}

float evalEpilogue(const EpGraph& g, float x,
                    const std::vector<const float*>& constBufs, int channelIdx)
{
    const std::vector<EpNode>& nodes = g.nodes();
    CV_Assert(g.outputNode >= 0 && g.outputNode < (int)nodes.size());
    CV_Assert(nodes.size() <= (size_t)EP_MAX_NODES);

    float v[EP_MAX_NODES];

    for (size_t i = 0; i < nodes.size(); i++) {
        const EpNode& n = nodes[i];
        const float a = n.inputs.size() > 0 ? v[n.inputs[0]] : 0.f;
        const float b = n.inputs.size() > 1 ? v[n.inputs[1]] : 0.f;

        switch (n.op) {
        case EpOP::INPUT: v[i] = x; break;
        case EpOP::CONST: v[i] = n.scalar; break;
        case EpOP::PER_CHANNEL_CONST:
            CV_DbgAssert(n.constBufferId >= 0 && n.constBufferId < (int)constBufs.size());
            v[i] = constBufs[n.constBufferId][channelIdx];
            break;
        case EpOP::ADD:  v[i] = a + b; break;
        case EpOP::SUB:  v[i] = a - b; break;
        case EpOP::MUL:  v[i] = a * b; break;
        case EpOP::MAX:  v[i] = a < b ? b : a; break;
        case EpOP::MIN:  v[i] = b < a ? b : a; break;
        case EpOP::ERF:  v[i] = std::erf(a); break;
        case EpOP::TANH: v[i] = std::tanh(a); break;
        case EpOP::EXP:  v[i] = std::exp(a); break;
        case EpOP::SQRT:  v[i] = std::sqrt(a); break;
        case EpOP::RECIP: v[i] = 1.f / a; break;
        case EpOP::CLAMP:
            v[i] = a < n.scalar ? n.scalar : (a > n.scalar2 ? n.scalar2 : a);
            break;
        }
    }
    return v[g.outputNode];
}

}// namespace dnn
}// namespace cv
