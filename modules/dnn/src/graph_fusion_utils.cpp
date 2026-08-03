// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "graph_fusion_utils.hpp"
#include <atomic>
#include <cfloat>
#include <unordered_set>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

static std::atomic<bool>& epilogueFusionFlag()
{
    static std::atomic<bool> enabled(!utils::getConfigurationParameterBool("OPENCV_DNN_DISABLE_EPILOGUE_FUSION", false));
    return enabled;
}

bool getEpilogueFusionEnabled() { return epilogueFusionFlag().load(); }

void setEpilogueFusionEnabled(bool enabled) { epilogueFusionFlag().store(enabled); }

bool epilogueDumpEnabled()
{
    static const bool on = utils::getConfigurationParameterBool("OPENCV_DNN_EPILOGUE_DUMP", false);
    return on;
}

static std::atomic<bool>& epilogueInterpFlag()
{
    static std::atomic<bool> on(utils::getConfigurationParameterBool("OPENCV_DNN_EPILOGUE_INTERP", false));
    return on;
}

bool epilogueInterpEnabled() { return epilogueInterpFlag().load(); }

void setEpilogueInterpEnabled(bool enabled) { epilogueInterpFlag().store(enabled); }

OpShape classify(const std::string& opType)
{
    static const std::unordered_set<std::string> anchors = {"Conv", "Gemm", "MatMul", "ConvTranspose"};
    static const std::unordered_set<std::string> reduces = {"Softmax", "BatchNormalization", "LayerNormalization", "ReduceMean"};
    static const std::unordered_set<std::string> maps    = {"Relu", "Sigmoid", "Tanh", "Erf", "Gelu", "Add", "Mul", "Sub", "Clip", "Max", "Min", "Sqrt", "Exp"};

    if (anchors.count(opType)) return OpShape::ANCHOR_TEMPLATE;
    if (reduces.count(opType)) return OpShape::ANCHOR_REDUCE;
    if (maps.count(opType))    return OpShape::MAP;
    return OpShape::UNCLASSIFIED;
}

static std::string naryOpType(const Ptr<LayerInfo>& l)
{
    Ptr<NaryEltwiseLayer> ew = l.dynamicCast<NaryEltwiseLayer>();
    if (!ew) return l->type;

    switch (ew->op) {
        case NaryEltwiseLayer::OPERATION::ADD:  return "Add";
        case NaryEltwiseLayer::OPERATION::SUB:  return "Sub";
        case NaryEltwiseLayer::OPERATION::PROD: return "Mul";
        case NaryEltwiseLayer::OPERATION::MAX:  return "Max";
        case NaryEltwiseLayer::OPERATION::MIN:  return "Min";
        default: return l->type;
    }
}

static std::string reluOpType(const Ptr<LayerInfo>& l)
{
    Ptr<ReLULayer> r = l.dynamicCast<ReLULayer>();
    if (r && r->negativeSlope != 0.f)
        return "LeakyRelu";
    return "Relu";
}

std::string effectiveOpType(const Ptr<LayerInfo>& l)
{
    const std::string& t = l->type;
    if (t == "NaryEltwise")         return naryOpType(l);
    if (t == "ReLU")                return reluOpType(l);
    if (t == "Conv2")               return "Conv";
    if (t == "ConvTranspose2")      return "ConvTranspose";
    if (t == "TanH")                return "Tanh";
    if (t == "BatchNorm2")          return "BatchNormalization";
    if (t == "LayerNormalization2") return "LayerNormalization";
    if (t == "Reduce2")             return "ReduceMean";
    return t;
}

void buildProducerOf(const Ptr<Graph>& g, int nargs, std::vector<int>& producerOf)
{
    producerOf.assign(nargs, -1);
    if (!g) return;

    const std::vector<Ptr<LayerInfo> >& prog = g->prog();
    for (size_t i = 0; i < prog.size(); i++) {
        if (!prog[i]) continue;
        for (Arg out : prog[i]->outputs) {
            if (out.idx > 0)
                producerOf[out.idx] = (int)i;
        }
    }
}

static int consumerOf(const std::vector<Ptr<LayerInfo> >& prog, Arg a, int from)
{
    for (size_t j = (size_t)std::max(from, 0); j < prog.size(); j++) {
        if (!prog[j]) continue;
        for (Arg in : prog[j]->inputs) {
            if (in.idx == a.idx) return (int)j;
        }
    }
    return -1;
}

static int bufIdFor(PointwiseChain& ch, Arg a)
{
    for (size_t k = 0; k < ch.constArgs.size(); k++) {
        if (ch.constArgs[k].idx == a.idx) return (int)k;
    }
    ch.constArgs.push_back(a);
    return (int)ch.constArgs.size() - 1;
}

static bool resolveOperand(const Ptr<LayerInfo>& L, const std::string& opType, Arg cur,
                           const ConstInfoFn& constInfo, PointwiseChain& ch, EpOperand& oper)
{
    if (opType == "Clip") {
        oper.scalar  = -FLT_MAX;
        oper.scalar2 =  FLT_MAX;
        ClipLayer* clip = dynamic_cast<ClipLayer*>(L.get());
        if (clip) {
            if (clip->hasMin) oper.scalar  = clip->minValue;
            if (clip->hasMax) oper.scalar2 = clip->maxValue;
        }
        for (size_t k = 1; k < L->inputs.size() && k <= 2; k++) {
            Arg a = L->inputs[k];
            if (a.idx == 0) continue;
            EpConstInfo info;
            if (!constInfo(a, info) || !info.isScalar) return false;
            if (k == 1) oper.scalar = info.scalar;
            else        oper.scalar2 = info.scalar;
        }
        return true;
    }

    Arg side(0);
    int nSide = 0;
    for (Arg in : L->inputs) {
        if (in.idx == cur.idx || in.idx == 0) continue;
        side = in;
        nSide++;
    }
    if (nSide == 0) return true;
    if (nSide > 1)  return false;

    if (opType == "Sub" && (L->inputs.empty() || L->inputs[0].idx != cur.idx))
        return false;

    EpConstInfo info;
    if (!constInfo(side, info)) return false;

    oper.hasSide = true;
    if (info.isScalar) {
        oper.scalar = info.scalar;
    } else {
        oper.perChannel = true;
        oper.bufId = bufIdFor(ch, side);
    }
    return true;
}

void collectPointwiseChains(const Ptr<Graph>& g, int nargs,
                             const std::vector<int>& useCounts,
                             const ConstInfoFn& constInfo,
                             std::vector<PointwiseChain>& chains)
{
    chains.clear();
    if (!g) return;
    CV_Assert((int)useCounts.size() == nargs);

    std::vector<int> producerOf;
    buildProducerOf(g, nargs, producerOf);

    const std::vector<Ptr<LayerInfo> >& prog = g->prog();
    std::vector<bool> taken(prog.size(), false);

    for (size_t i = 0; i < prog.size(); i++) {
        if (!prog[i] || taken[i]) continue;
        if (classify(effectiveOpType(prog[i])) != OpShape::ANCHOR_TEMPLATE) continue;
        if (prog[i]->outputs.size() != 1) continue;

        PointwiseChain ch;
        ch.nodes.push_back((int)i);

        EpBuilder b;
        int epCur = b.push(EpOP::INPUT, {});
        bool epOpen = true;
        Arg curArg = prog[i]->outputs[0];

        while (curArg.idx > 0 && curArg.idx < (int)useCounts.size() && useCounts[curArg.idx] == 1) {
            int j = consumerOf(prog, curArg, producerOf[curArg.idx] + 1);
            if (j < 0 || taken[j]) break;

            const Ptr<LayerInfo>& L = prog[j];
            const std::string opType = effectiveOpType(L);
            const bool isActiv = dynamic_cast<const ActivationLayer*>(L.get()) != nullptr;
            const bool isMap = classify(opType) == OpShape::MAP;
            if (!isActiv && !isMap) break;
            if (L->outputs.size() != 1) break;
            if (L->subgraphs()) break;

            bool appended = false;
            EpOperand oper;
            if (epOpen && isMap) {
                if (resolveOperand(L, opType, curArg, constInfo, ch, oper)) {
                    int next = appendNode(b, epCur, opType, oper);
                    if (next >= 0) {
                        epCur = next;
                        ch.epSteps = (int)ch.absorbed.size() + 1;
                        appended = true;
                    }
                }
            }
            if (!appended) {
                if (!isActiv || L->inputs.size() != 1) break;
                epOpen = false;
                oper = EpOperand();
            }

            ch.nodes.push_back(j);
            ch.absorbed.push_back(L);
            ch.stepOperands.push_back(oper);
            curArg = L->outputs[0];
        }

        if (ch.nodes.size() > 1) {
            if (ch.epSteps > 0) {
                b.g.outputNode = epCur;
                ch.ep = b.g;
            }
            for (int n : ch.nodes) taken[n] = true;
            chains.push_back(ch);
        }
    }
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
