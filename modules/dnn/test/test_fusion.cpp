// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>
#include "../src/graph_fusion_utils.hpp"
#include "../src/layers/cpu_kernels/fusion_apply.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace opencv_test { namespace {

using namespace cv::dnn;
using std::vector;

static FusionOperand perChannelOp(int bufId)
{
    FusionOperand o;
    o.hasSide = true;
    o.perChannel = true;
    o.bufId = bufId;
    return o;
}

static FusionOperand scalarOp(float v)
{
    FusionOperand o;
    o.hasSide = true;
    o.scalar = v;
    return o;
}

static FusionOperand clipOp(float lo, float hi)
{
    FusionOperand o;
    o.scalar = lo;
    o.scalar2 = hi;
    return o;
}

static FusionGraph build(const vector<std::pair<std::string, FusionOperand> >& ops)
{
    FusionGraphBuilder b;
    int cur = b.push(FusionOp::INPUT, {});
    for (size_t i = 0; i < ops.size(); i++) {
        cur = appendFusionNode(b, cur, ops[i].first, ops[i].second);
        CV_Assert(cur >= 0);
    }
    b.g.outputNode = cur;
    return b.g;
}

static vector<FusionOp> opsOf(const FusionGraph& fg)
{
    vector<FusionOp> r;
    for (const FusionNode& n : fg.nodes()) r.push_back(n.op);
    return r;
}

static float eval(const FusionGraph& g, float x)
{
    return evalFusionGraph(g, x, vector<const float*>(), 0);
}


TEST(FusionGraph, CommutativeOpsAreCanonicalised)
{
    FusionGraphBuilder b1;
    int x1 = b1.push(FusionOp::INPUT, {});
    int c1 = b1.push(FusionOp::CONST, {}, 2.f);
    int a1 = b1.push(FusionOp::ADD, {x1, c1});

    FusionGraphBuilder b2;
    int x2 = b2.push(FusionOp::INPUT, {});
    int c2 = b2.push(FusionOp::CONST, {}, 2.f);
    int a2 = b2.push(FusionOp::ADD, {c2, x2});

    EXPECT_EQ(b1.g.nodes()[a1].inputs, b2.g.nodes()[a2].inputs);
}

TEST(FusionGraph, SubIsNotReordered)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int c = b.push(FusionOp::CONST, {}, 2.f);
    int s = b.push(FusionOp::SUB, {c, x});
    EXPECT_EQ(b.g.nodes()[s].inputs, vector<int>({c, x}));
}

TEST(FusionGraph, MaxIsNotReordered)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int c = b.push(FusionOp::CONST, {}, 2.f);
    int m = b.push(FusionOp::MAX, {c, x});
    EXPECT_EQ(b.g.nodes()[m].inputs, vector<int>({c, x}));
}

TEST(FusionGraph, HashConsingDedupesRepeatedSubexpression)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int c = b.push(FusionOp::CONST, {}, 2.f);
    int m1 = b.push(FusionOp::MUL, {x, c});
    int m2 = b.push(FusionOp::MUL, {x, c});
    EXPECT_EQ(m1, m2);
    EXPECT_EQ(b.g.size(), 3u);
}

TEST(FusionGraph, DistinctConstantsDoNotMerge)
{
    FusionGraphBuilder b;
    b.push(FusionOp::INPUT, {});
    int a = b.push(FusionOp::CONST, {}, 1.f);
    int c = b.push(FusionOp::CONST, {}, 2.f);
    EXPECT_NE(a, c);
}

TEST(FusionGraph, SignedZeroDoesNotMerge)
{
    FusionGraphBuilder b;
    b.push(FusionOp::INPUT, {});
    int p = b.push(FusionOp::CONST, {},  0.f);
    int n = b.push(FusionOp::CONST, {}, -0.f);
    EXPECT_NE(p, n);
}

TEST(FusionGraph, FirstNodeMustBeInput)
{
    FusionGraphBuilder b;
    EXPECT_ANY_THROW(b.push(FusionOp::CONST, {}, 1.f));
}

TEST(FusionGraph, ArityMismatchThrows)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    EXPECT_ANY_THROW(b.push(FusionOp::ADD, {x}));
    EXPECT_ANY_THROW(b.push(FusionOp::SQRT, {x, x}));
}


TEST(AppendFusionNode, ReluIsMaxAgainstZero)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int y = appendFusionNode(b, x, "Relu", FusionOperand());

    ASSERT_GE(y, 0);
    EXPECT_EQ(opsOf(b.g), vector<FusionOp>({FusionOp::INPUT, FusionOp::CONST, FusionOp::MAX}));
    EXPECT_EQ(b.g.nodes()[1].scalar, 0.f);
}

TEST(AppendFusionNode, SigmoidUsesReciprocalOfExp)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int y = appendFusionNode(b, x, "Sigmoid", FusionOperand());
    ASSERT_GE(y, 0);

    vector<FusionOp> ops = opsOf(b.g);
    EXPECT_EQ(std::count(ops.begin(), ops.end(), FusionOp::EXP),   1);
    EXPECT_EQ(std::count(ops.begin(), ops.end(), FusionOp::RECIP), 1);
    EXPECT_EQ(std::count(ops.begin(), ops.end(), FusionOp::TANH),  0);
    EXPECT_EQ(ops[y], FusionOp::RECIP);
}

TEST(AppendFusionNode, GeluDecomposesToErf)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int y = appendFusionNode(b, x, "Gelu", FusionOperand());

    ASSERT_GE(y, 0);
    vector<FusionOp> ops = opsOf(b.g);
    EXPECT_EQ(std::count(ops.begin(), ops.end(), FusionOp::ERF), 1);
    EXPECT_EQ(ops[y], FusionOp::MUL);
}

TEST(AppendFusionNode, ClampCarriesBothBounds)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int y = appendFusionNode(b, x, "Clip", clipOp(-1.f, 6.f));

    ASSERT_GE(y, 0);
    EXPECT_EQ(b.g.nodes()[y].op, FusionOp::CLAMP);
    EXPECT_EQ(b.g.nodes()[y].scalar,  -1.f);
    EXPECT_EQ(b.g.nodes()[y].scalar2,  6.f);
}

TEST(AppendFusionNode, PerChannelSideOperand)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int y = appendFusionNode(b, x, "Add", perChannelOp(3));

    ASSERT_GE(y, 0);
    EXPECT_EQ(opsOf(b.g), vector<FusionOp>({FusionOp::INPUT, FusionOp::PER_CHANNEL_CONST, FusionOp::ADD}));
    EXPECT_EQ(b.g.nodes()[1].constBufferId, 3);
}

TEST(AppendFusionNode, BinaryWithoutSideOperandIsRejected)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    EXPECT_LT(appendFusionNode(b, x, "Add", FusionOperand()), 0);
    EXPECT_LT(appendFusionNode(b, x, "Mul", FusionOperand()), 0);
}

TEST(AppendFusionNode, UnknownOpIsRejected)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    EXPECT_LT(appendFusionNode(b, x, "Gather", FusionOperand()), 0);
    EXPECT_LT(appendFusionNode(b, x, "",       FusionOperand()), 0);
}


TEST(FusionInterpret, Relu)
{
    FusionGraph g = build({{"Relu", FusionOperand()}});
    EXPECT_FLOAT_EQ(eval(g,  2.5f), 2.5f);
    EXPECT_FLOAT_EQ(eval(g, -2.5f), 0.f);
}

TEST(FusionInterpret, ClampUsesBothBounds)
{
    FusionGraph g = build({{"Clip", clipOp(-1.f, 6.f)}});
    EXPECT_FLOAT_EQ(eval(g, -9.f), -1.f);
    EXPECT_FLOAT_EQ(eval(g,  3.f),  3.f);
    EXPECT_FLOAT_EQ(eval(g,  9.f),  6.f);
}

TEST(FusionInterpret, PerChannelBiasReadsTheRightChannel)
{
    FusionGraph g = build({{"Add", perChannelOp(0)}});
    const float bias[3] = {10.f, 20.f, 30.f};
    vector<const float*> bufs(1, bias);

    EXPECT_FLOAT_EQ(evalFusionGraph(g, 1.f, bufs, 0), 11.f);
    EXPECT_FLOAT_EQ(evalFusionGraph(g, 1.f, bufs, 2), 31.f);
}

TEST(FusionInterpret, SigmoidMatchesReference)
{
    FusionGraph g = build({{"Sigmoid", FusionOperand()}});
    for (float x = -6.f; x <= 6.f; x += 0.5f)
        EXPECT_NEAR(eval(g, x), 1.f / (1.f + std::exp(-x)), 1e-6) << "x=" << x;
}

TEST(FusionInterpret, GeluMatchesErfReference)
{
    FusionGraph g = build({{"Gelu", FusionOperand()}});
    for (float x = -6.f; x <= 6.f; x += 0.5f) {
        const float ref = 0.5f * x * (1.f + std::erf(x / std::sqrt(2.f)));
        EXPECT_NEAR(eval(g, x), ref, 1e-6) << "x=" << x;
    }
}

TEST(FusionInterpret, ChainEvaluatesInOrder)
{
    FusionGraph g = build({{"Mul", scalarOp(3.f)}, {"Add", scalarOp(1.f)}, {"Sqrt", FusionOperand()}});
    EXPECT_FLOAT_EQ(eval(g, 5.f), 4.f);
}

TEST(FusionInterpret, SharedSubexpressionEvaluatedOnce)
{
    FusionGraphBuilder b;
    int x = b.push(FusionOp::INPUT, {});
    int t = b.push(FusionOp::MUL, {x, b.push(FusionOp::CONST, {}, 2.f)});
    b.g.outputNode = b.push(FusionOp::ADD, {t, t});

    EXPECT_EQ(b.g.size(), 4u);
    EXPECT_FLOAT_EQ(eval(b.g, 3.f), 12.f);
}


class MockGraph : public Graph
{
public:
    bool empty() const override { return prog_.empty(); }
    void clear() override { prog_.clear(); }
    std::string name() const override { return "mock"; }

    const vector<Arg>& append(Ptr<LayerInfo>&, const vector<std::string>&) override
    { CV_Error(Error::StsNotImplemented, ""); }
    Arg append(Ptr<LayerInfo>&, const std::string&) override
    { CV_Error(Error::StsNotImplemented, ""); }
    std::ostream& dump(std::ostream& strm, int, bool) override { return strm; }

    const vector<Arg>& inputs() const override { return inputs_; }
    const vector<Arg>& outputs() const override { return outputs_; }
    void setOutputs(const vector<Arg>& outs) override { outputs_ = outs; }
    const vector<Ptr<LayerInfo> >& prog() const override { return prog_; }
    void setProg(const vector<Ptr<LayerInfo> >& p) override { prog_ = p; }
    int opBackend(int) const override { return DNN_BACKEND_OPENCV; }

    vector<Arg> inputs_, outputs_;
    vector<Ptr<LayerInfo> > prog_;
};

static Ptr<Layer> mkLayer(const String& type, const vector<int>& ins, const vector<int>& outs)
{
    Ptr<Layer> l = makePtr<Layer>();
    l->type = type;
    for (int i : ins)  l->inputs.push_back(Arg(i));
    for (int o : outs) l->outputs.push_back(Arg(o));
    return l;
}

static Ptr<MockGraph> mkGraph(const vector<Ptr<LayerInfo> >& prog, const vector<int>& gOuts)
{
    Ptr<MockGraph> g = makePtr<MockGraph>();
    g->setProg(prog);
    vector<Arg> outs;
    for (int o : gOuts) outs.push_back(Arg(o));
    g->setOutputs(outs);
    return g;
}

static vector<int> mkUseCounts(const Ptr<MockGraph>& g, int nargs)
{
    vector<int> uc(nargs, 0);
    uc[0] = 1;
    for (Arg o : g->outputs()) uc[o.idx]++;
    for (const Ptr<LayerInfo>& l : g->prog())
        if (l) for (Arg in : l->inputs) uc[in.idx]++;
    return uc;
}

static ConstInfoFn noConsts()
{
    return [](Arg, FusionConstInfo&) { return false; };
}

static ConstInfoFn perChannelArgs(const std::set<int>& ids)
{
    return [ids](Arg a, FusionConstInfo& info) {
        if (!ids.count(a.idx)) return false;
        info.isScalar = false;
        return true;
    };
}

static ConstInfoFn scalarArgs(const std::map<int, float>& vals)
{
    return [vals](Arg a, FusionConstInfo& info) {
        std::map<int, float>::const_iterator it = vals.find(a.idx);
        if (it == vals.end()) return false;
        info.isScalar = true;
        info.scalar = it->second;
        return true;
    };
}

TEST(GraphFusionUtils, ClassifyKnownOps)
{
    EXPECT_EQ(classify("Conv"),               OpShape::ANCHOR_TEMPLATE);
    EXPECT_EQ(classify("Gemm"),               OpShape::ANCHOR_TEMPLATE);
    EXPECT_EQ(classify("MatMul"),             OpShape::ANCHOR_TEMPLATE);
    EXPECT_EQ(classify("ConvTranspose"),      OpShape::ANCHOR_TEMPLATE);
    EXPECT_EQ(classify("Softmax"),            OpShape::ANCHOR_REDUCE);
    EXPECT_EQ(classify("LayerNormalization"), OpShape::ANCHOR_REDUCE);
    EXPECT_EQ(classify("Relu"),               OpShape::MAP);
    EXPECT_EQ(classify("Clip"),               OpShape::MAP);
    EXPECT_EQ(classify("Add"),                OpShape::MAP);
}

TEST(GraphFusionUtils, ClassifyUnknownIsUnclassified)
{
    EXPECT_EQ(classify("NonMaxSuppression"), OpShape::UNCLASSIFIED);
    EXPECT_EQ(classify("Gather"),            OpShape::UNCLASSIFIED);
    EXPECT_EQ(classify("If"),                OpShape::UNCLASSIFIED);
    EXPECT_EQ(classify(""),                  OpShape::UNCLASSIFIED);
    EXPECT_EQ(classify("relu"),              OpShape::UNCLASSIFIED);
    EXPECT_EQ(classify("Pooling"),           OpShape::UNCLASSIFIED);
}

TEST(GraphFusionUtils, EffectiveOpTypeNormalisesInternalNames)
{
    EXPECT_EQ(effectiveOpType(mkLayer("Conv2", {}, {})),               "Conv");
    EXPECT_EQ(effectiveOpType(mkLayer("ConvTranspose2", {}, {})),      "ConvTranspose");
    EXPECT_EQ(effectiveOpType(mkLayer("TanH", {}, {})),                "Tanh");
    EXPECT_EQ(effectiveOpType(mkLayer("BatchNorm2", {}, {})),          "BatchNormalization");
    EXPECT_EQ(effectiveOpType(mkLayer("LayerNormalization2", {}, {})), "LayerNormalization");
    EXPECT_EQ(effectiveOpType(mkLayer("Reduce2", {}, {})),             "ReduceMean");
    EXPECT_EQ(effectiveOpType(mkLayer("Gemm", {}, {})),                "Gemm");
}

TEST(GraphFusionUtils, LeakyReluIsNotNormalisedToRelu)
{
    LayerParams plain;
    Ptr<Layer> relu = ReLULayer::create(plain);
    relu->type = "ReLU";
    EXPECT_EQ(effectiveOpType(relu), "Relu");
    EXPECT_EQ(classify(effectiveOpType(relu)), OpShape::MAP);

    LayerParams leakyParams;
    leakyParams.set("negative_slope", 0.1f);
    Ptr<Layer> leaky = ReLULayer::create(leakyParams);
    leaky->type = "ReLU";
    EXPECT_EQ(effectiveOpType(leaky), "LeakyRelu");
    EXPECT_EQ(classify(effectiveOpType(leaky)), OpShape::UNCLASSIFIED);
}

static Ptr<Layer> mkRealActiv(const Ptr<Layer>& l, const String& type, int in, int out)
{
    l->type = type;
    l->inputs.push_back(Arg(in));
    l->outputs.push_back(Arg(out));
    return l;
}

static Ptr<Layer> mkLeaky(int in, int out)
{
    LayerParams lp;
    lp.set("negative_slope", 0.1f);
    return mkRealActiv(ReLULayer::create(lp), "ReLU", in, out);
}

TEST(GraphFusionUtils, LeakyReluIsAbsorbedWithoutIrStep)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}), mkLeaky(3, 4)}, {4});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].nodes, vector<int>({0, 1}));
    EXPECT_EQ(chains[0].fgSteps, 0);
    EXPECT_TRUE(chains[0].fg.empty());
}

TEST(GraphFusionUtils, ActivationOutsideMapSetIsAbsorbed)
{
    LayerParams lp;
    Ptr<Layer> swish = mkRealActiv(SwishLayer::create(lp), "Swish", 3, 4);
    ASSERT_EQ(classify(effectiveOpType(swish)), OpShape::UNCLASSIFIED);

    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}), swish}, {4});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].absorbed.size(), 1u);
    EXPECT_EQ(chains[0].fgSteps, 0);
}

TEST(GraphFusionUtils, IrPrefixStopsAtFirstOpaqueStep)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("Add",  {3, 9}, {4}),
                                mkLeaky(4, 5)}, {5});
    vector<int> uc = mkUseCounts(g, 12);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 12, uc, scalarArgs({{9, 2.f}}), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].absorbed.size(), 2u);
    EXPECT_EQ(chains[0].fgSteps, 1);
    EXPECT_NEAR(eval(chains[0].fg, 1.f), 3.f, 1e-6);
}

TEST(GraphFusionUtils, IrPrefixNeverResumesAfterOpaqueStep)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLeaky(3, 4),
                                mkLayer("Add", {4, 9}, {5})}, {5});
    vector<int> uc = mkUseCounts(g, 12);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 12, uc, scalarArgs({{9, 2.f}}), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].absorbed.size(), 1u);
    EXPECT_EQ(chains[0].fgSteps, 0);
}

TEST(GraphFusionUtils, ProducerOfMapsArgToProgIndex)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("ReLU", {3},    {4})}, {4});
    vector<int> p;
    buildProducerOf(g, 8, p);

    EXPECT_EQ((int)p.size(), 8);
    EXPECT_EQ(p[3], 0);
    EXPECT_EQ(p[4], 1);
    EXPECT_EQ(p[1], -1);
}

TEST(GraphFusionUtils, ProducerOfIgnoresSentinelArg)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1}, {0, 3})}, {3});
    vector<int> p;
    buildProducerOf(g, 8, p);

    EXPECT_EQ(p[0], -1);
    EXPECT_EQ(p[3], 0);
}

TEST(GraphFusionUtils, ProducerOfHandlesNullGraphAndNullLayers)
{
    vector<int> p;
    buildProducerOf(Ptr<Graph>(), 4, p);
    EXPECT_EQ((int)p.size(), 4);
    EXPECT_EQ(p[3], -1);

    Ptr<MockGraph> g = mkGraph({Ptr<Layer>(), mkLayer("Conv2", {1}, {3})}, {3});
    buildProducerOf(g, 8, p);
    EXPECT_EQ(p[3], 1);
}

TEST(GraphFusionUtils, CollectAbsorbsBiasAdd)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("Add",  {3, 4}, {5})}, {5});
    vector<int> uc = mkUseCounts(g, 8);
    ASSERT_EQ(uc[3], 1);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, perChannelArgs({4}), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].nodes, vector<int>({0, 1}));
    EXPECT_EQ(opsOf(chains[0].fg), vector<FusionOp>({FusionOp::INPUT, FusionOp::PER_CHANNEL_CONST, FusionOp::ADD}));
    EXPECT_EQ(chains[0].fg.outputNode, 2);
    ASSERT_EQ(chains[0].constArgs.size(), 1u);
    EXPECT_EQ(chains[0].constArgs[0].idx, 4);
}

TEST(GraphFusionUtils, CollectAbsorbsMultiOpChain)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("ReLU", {3},    {4}),
                                mkLayer("Add",  {4, 9}, {5}),
                                mkLayer("Clip", {5},    {6})}, {6});
    vector<int> uc = mkUseCounts(g, 12);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 12, uc, perChannelArgs({9}), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].nodes, vector<int>({0, 1, 2, 3}));
    EXPECT_EQ(chains[0].fg.nodes()[chains[0].fg.outputNode].op, FusionOp::CLAMP);
}

TEST(GraphFusionUtils, CollectUsesScalarConstWhenOperandIsScalar)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("Mul",  {3, 4}, {5})}, {5});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, scalarArgs({{4, 2.5f}}), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(opsOf(chains[0].fg), vector<FusionOp>({FusionOp::INPUT, FusionOp::CONST, FusionOp::MUL}));
    EXPECT_EQ(chains[0].fg.nodes()[1].scalar, 2.5f);
    EXPECT_TRUE(chains[0].constArgs.empty());
}

TEST(GraphFusionUtils, CollectStopsOnFanOut)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("ReLU", {3},    {4}),
                                mkLayer("Sqrt", {3},    {5})}, {4, 5});
    vector<int> uc = mkUseCounts(g, 8);
    ASSERT_EQ(uc[3], 2);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);
    EXPECT_TRUE(chains.empty());
}

TEST(GraphFusionUtils, CollectStopsWhenAnchorOutputIsGraphOutput)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("ReLU", {3},    {4})}, {3, 4});
    vector<int> uc = mkUseCounts(g, 8);
    ASSERT_EQ(uc[3], 2);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);
    EXPECT_TRUE(chains.empty());
}

TEST(GraphFusionUtils, CollectStopsOnDynamicSecondInput)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("ReLU", {3},    {4}),
                                mkLayer("Add",  {4, 9}, {5})}, {5});
    vector<int> uc = mkUseCounts(g, 12);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 12, uc, noConsts(), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].nodes, vector<int>({0, 1}));
}

TEST(GraphFusionUtils, CollectStopsOnSubWithConstOnLeft)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("ReLU", {3},    {4}),
                                mkLayer("Sub",  {9, 4}, {5})}, {5});
    vector<int> uc = mkUseCounts(g, 12);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 12, uc, scalarArgs({{9, 1.f}}), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].nodes, vector<int>({0, 1}));
}

TEST(GraphFusionUtils, CollectStopsOnNonMapOp)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2",    {1, 2}, {3}),
                                mkLayer("ReLU",    {3},    {4}),
                                mkLayer("Softmax", {4},    {5})}, {5});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);

    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].nodes, vector<int>({0, 1}));
}

TEST(GraphFusionUtils, CollectStopsOnMultiOutputConsumer)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("Clip", {3},    {4, 5})}, {4, 5});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);
    EXPECT_TRUE(chains.empty());
}

TEST(GraphFusionUtils, CollectIgnoresMultiOutputAnchor)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3, 7}),
                                mkLayer("ReLU", {3},    {4})}, {4, 7});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);
    EXPECT_TRUE(chains.empty());
}

TEST(GraphFusionUtils, CollectFindsNothingWithoutAnchor)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("ReLU", {1}, {2}),
                                mkLayer("Sqrt", {2}, {3})}, {3});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);
    EXPECT_TRUE(chains.empty());
}

TEST(GraphFusionUtils, CollectHandlesTwoIndependentAnchors)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("ReLU", {3},    {4}),
                                mkLayer("Gemm", {4, 5}, {6}),
                                mkLayer("TanH", {6},    {7})}, {7});
    vector<int> uc = mkUseCounts(g, 12);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 12, uc, noConsts(), chains);

    ASSERT_EQ(chains.size(), 2u);
    EXPECT_EQ(chains[0].nodes, vector<int>({0, 1}));
    EXPECT_EQ(chains[1].nodes, vector<int>({2, 3}));
}

TEST(GraphFusionUtils, GeluChainIsCollectedAndInterpretable)
{
    Ptr<MockGraph> g = mkGraph({mkLayer("Conv2", {1, 2}, {3}),
                                mkLayer("Gelu", {3},    {4})}, {4});
    vector<int> uc = mkUseCounts(g, 8);

    vector<AgnosticChain> chains;
    collectAgnosticChains(g, 8, uc, noConsts(), chains);

    ASSERT_EQ(chains.size(), 1u);

    const float ref = 0.5f * 1.5f * (1.f + std::erf(1.5f / std::sqrt(2.f)));
    EXPECT_NEAR(eval(chains[0].fg, 1.5f), ref, 1e-6);
}


static AgnosticChain mkChain(const vector<Ptr<LayerInfo> >& absorbed,
                              const vector<std::pair<std::string, FusionOperand> >& ops)
{
    AgnosticChain ch;
    ch.nodes.push_back(0);
    for (size_t i = 0; i < absorbed.size(); i++) {
        ch.absorbed.push_back(absorbed[i]);
        ch.nodes.push_back((int)i + 1);
        ch.stepOperands.push_back(i < ops.size() ? ops[i].second : FusionOperand());
    }
    ch.fg = build(ops);
    ch.fgSteps = (int)ops.size();
    return ch;
}

static Ptr<Layer> mkAdd()
{
    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.set("operation", "add");
    return NaryEltwiseLayer::create(lp);
}

static Ptr<Layer> mkClip()
{
    LayerParams lp;
    lp.type = "Clip";
    return ClipLayer::create(lp);
}

static FusionSink* sinkOf(const Ptr<Layer>& l)
{
    return dynamic_cast<FusionSink*>(l.get());
}

static FusionSink* sink(const Ptr<Layer>& l)
{
    FusionSink* s = sinkOf(l);
    CV_Assert(s != nullptr);
    return s;
}

static Ptr<Layer> mkConv()
{
    LayerParams lp;
    lp.set("kernel_size", 1);
    return Conv2Layer::create(lp);
}

static Ptr<Layer> mkGemm()
{
    LayerParams lp;
    return GemmLayer::create(lp);
}

static Ptr<Layer> mkMatMul()
{
    LayerParams lp;
    lp.type = "MatMul";
    return MatMulLayer::create(lp);
}

static Ptr<Layer> mkConvW(int Cout = 8, int Cin = 8)
{
    LayerParams lp;
    lp.set("kernel_size", 1);
    Ptr<Conv2Layer> c = Conv2Layer::create(lp);
    int wsz[4] = {Cout, Cin, 1, 1};
    Mat w(4, wsz, CV_32F, Scalar(0.1f));
    c->setWeights(w, Mat(), 8, -1);
    return c;
}

static Mat perChanConst(int C, int nspatial, float v)
{
    std::vector<int> sz(1 + nspatial, 1);
    sz[0] = C;
    return Mat((int)sz.size(), sz.data(), CV_32F, Scalar(v));
}

TEST(FusionLowering, ConvAndGemmAreSinks)
{
    EXPECT_NE(sinkOf(mkConv()), nullptr);
    EXPECT_NE(sinkOf(mkGemm()), nullptr);
    EXPECT_NE(sinkOf(mkMatMul()), nullptr);
}

TEST(FusionLowering, MatMulAcceptsActivationChain)
{
    LayerParams lp;
    Ptr<Layer> gelu = GeluLayer::create(lp);
    Ptr<Layer> tanh = TanHLayer::create(lp);

    Ptr<Layer> one = mkMatMul();
    EXPECT_TRUE(sink(one)->setFusion(mkChain({gelu}, {{"Gelu", FusionOperand()}})));

    Ptr<Layer> two = mkMatMul();
    EXPECT_TRUE(sink(two)->setFusion(
        mkChain({gelu, tanh}, {{"Gelu", FusionOperand()}, {"Tanh", FusionOperand()}})));
}

TEST(FusionLowering, MatMulRefusesWhenAlreadyFused)
{
    LayerParams lp;
    Ptr<Layer> gelu = GeluLayer::create(lp);
    Ptr<Layer> mm = mkMatMul();
    EXPECT_TRUE(sink(mm)->setFusion(mkChain({gelu}, {{"Gelu", FusionOperand()}})));
    EXPECT_FALSE(sink(mm)->setFusion(mkChain({gelu}, {{"Gelu", FusionOperand()}})));
}

TEST(FusionLowering, MatMulIsAnAnchor)
{
    EXPECT_EQ(classify("MatMul"), OpShape::ANCHOR_TEMPLATE);
    EXPECT_EQ(effectiveOpType(mkMatMul()), "MatMul");
}

TEST(FusionLowering, ConvAcceptsSingleRelu)
{
    Ptr<Layer> conv = mkConv();
    LayerParams lp;
    Ptr<Layer> relu = ReLULayer::create(lp);

    ASSERT_NE(sinkOf(conv), nullptr);
    EXPECT_TRUE(sink(conv)->setFusion(mkChain({relu}, {{"Relu", FusionOperand()}})));
}

TEST(FusionLowering, ConvRejectsSecondActivation)
{
    Ptr<Layer> conv = mkConv();
    LayerParams lp;
    Ptr<Layer> a = ReLULayer::create(lp);
    Ptr<Layer> b = TanHLayer::create(lp);

    EXPECT_FALSE(sink(conv)->setFusion(
        mkChain({a, b}, {{"Relu", FusionOperand()}, {"Tanh", FusionOperand()}})));
}

TEST(FusionLowering, ConvRejectsClampWithNonZeroLowerBound)
{
    LayerParams lp;
    Ptr<Layer> clip = ClipLayer::create(lp);

    Ptr<Layer> ok = mkConv();
    EXPECT_TRUE(sink(ok)->setFusion(mkChain({clip}, {{"Clip", clipOp(0.f, 6.f)}})));

    Ptr<Layer> bad = mkConv();
    EXPECT_FALSE(sink(bad)->setFusion(mkChain({clip}, {{"Clip", clipOp(-1.f, 6.f)}})));
}

TEST(FusionLowering, ConvFoldsPerChannelAddIntoBias)
{
    AgnosticChain ch = mkChain({mkAdd()}, {{"Add", perChannelOp(0)}});
    ch.constBufs.push_back(perChanConst(8, 2, 0.25f));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_TRUE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvFoldsScalarAddIntoBias)
{
    Ptr<Layer> conv = mkConvW(8);
    EXPECT_TRUE(sink(conv)->setFusion(mkChain({mkAdd()}, {{"Add", scalarOp(0.5f)}})));
}

TEST(FusionLowering, ConvRejectsAddOnWrongAxis)
{
    int sz[4] = {1, 1, 1, 8};
    AgnosticChain ch = mkChain({mkAdd()}, {{"Add", perChannelOp(0)}});
    ch.constBufs.push_back(Mat(4, sz, CV_32F, Scalar(0.25f)));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_FALSE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvRejectsAddWithWrongChannelCount)
{
    AgnosticChain ch = mkChain({mkAdd()}, {{"Add", perChannelOp(0)}});
    ch.constBufs.push_back(perChanConst(4, 2, 0.25f));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_FALSE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvRejectsAddBeforeWeightsAreSet)
{
    AgnosticChain ch = mkChain({mkAdd()}, {{"Add", perChannelOp(0)}});
    ch.constBufs.push_back(perChanConst(8, 2, 0.25f));

    Ptr<Layer> conv = mkConv();
    EXPECT_FALSE(sink(conv)->setFusion(ch));
}

static Ptr<Layer> mkMul()
{
    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.set("operation", "mul");
    return NaryEltwiseLayer::create(lp);
}

TEST(FusionLowering, ConvFoldsMulAddClipChain)
{
    AgnosticChain ch = mkChain({mkMul(), mkAdd(), mkClip()},
                                {{"Mul", perChannelOp(0)},
                                 {"Add", perChannelOp(1)},
                                 {"Clip", clipOp(0.f, 6.f)}});
    ch.constBufs.push_back(perChanConst(8, 2, 2.0f));
    ch.constBufs.push_back(perChanConst(8, 2, 0.5f));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_TRUE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvFoldsMulThenAdd)
{
    AgnosticChain ch = mkChain({mkMul(), mkAdd()},
                                {{"Mul", perChannelOp(0)}, {"Add", perChannelOp(1)}});
    ch.constBufs.push_back(perChanConst(8, 2, 2.0f));
    ch.constBufs.push_back(perChanConst(8, 2, 0.5f));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_TRUE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvRejectsAddBeforeMul)
{
    AgnosticChain ch = mkChain({mkAdd(), mkMul()},
                                {{"Add", perChannelOp(0)}, {"Mul", perChannelOp(1)}});
    ch.constBufs.push_back(perChanConst(8, 2, 0.5f));
    ch.constBufs.push_back(perChanConst(8, 2, 2.0f));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_FALSE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvRejectsAffineWithTrailingNonActivation)
{
    AgnosticChain ch = mkChain({mkMul(), mkClip(), mkAdd()},
                                {{"Mul", perChannelOp(0)},
                                 {"Clip", clipOp(0.f, 6.f)},
                                 {"Add", perChannelOp(1)}});
    ch.constBufs.push_back(perChanConst(8, 2, 2.0f));
    ch.constBufs.push_back(perChanConst(8, 2, 0.5f));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_FALSE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvRejectsAffineOnWrongAxis)
{
    int sz[4] = {1, 1, 1, 8};
    AgnosticChain ch = mkChain({mkMul(), mkAdd()},
                                {{"Mul", perChannelOp(0)}, {"Add", perChannelOp(1)}});
    ch.constBufs.push_back(Mat(4, sz, CV_32F, Scalar(2.0f)));
    ch.constBufs.push_back(perChanConst(8, 2, 0.5f));

    Ptr<Layer> conv = mkConvW(8);
    EXPECT_FALSE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, ConvAffineLeavesStateCleanOnRefusal)
{
    AgnosticChain bad = mkChain({mkAdd(), mkMul()},
                                 {{"Add", perChannelOp(0)}, {"Mul", perChannelOp(1)}});
    bad.constBufs.push_back(perChanConst(8, 2, 0.5f));
    bad.constBufs.push_back(perChanConst(8, 2, 2.0f));

    Ptr<Layer> conv = mkConvW(8);
    ASSERT_FALSE(sink(conv)->setFusion(bad));

    AgnosticChain good = mkChain({mkMul(), mkAdd()},
                                  {{"Mul", perChannelOp(0)}, {"Add", perChannelOp(1)}});
    good.constBufs.push_back(perChanConst(8, 2, 2.0f));
    good.constBufs.push_back(perChanConst(8, 2, 0.5f));
    EXPECT_TRUE(sink(conv)->setFusion(good));
}

TEST(FusionLowering, ConvRejectsAddAfterResidualFold)
{
    AgnosticChain ch = mkChain({mkAdd()}, {{"Add", perChannelOp(0)}});
    ch.constBufs.push_back(perChanConst(8, 2, 0.25f));

    Ptr<Layer> conv = mkConvW(8);
    ASSERT_TRUE(conv.dynamicCast<Conv2Layer>()->fuseAddResidual(Arg(5)));
    EXPECT_FALSE(sink(conv)->setFusion(ch));
}

TEST(FusionLowering, GemmAcceptsActivationChain)
{
    LayerParams lp;
    Ptr<Layer> gelu = GeluLayer::create(lp);
    Ptr<Layer> tanh = TanHLayer::create(lp);

    Ptr<Layer> one = mkGemm();
    EXPECT_TRUE(sink(one)->setFusion(mkChain({gelu}, {{"Gelu", FusionOperand()}})));

    Ptr<Layer> two = mkGemm();
    EXPECT_TRUE(sink(two)->setFusion(
        mkChain({gelu, tanh}, {{"Gelu", FusionOperand()}, {"Tanh", FusionOperand()}})));
}

TEST(FusionLowering, GemmAcceptsScalarAdd)
{
    Ptr<Layer> gemm = mkGemm();
    EXPECT_TRUE(sink(gemm)->setFusion(mkChain({mkAdd()}, {{"Add", scalarOp(1.f)}})));
}

TEST(FusionLowering, GemmRejectsAddWithoutConstOperand)
{
    Ptr<Layer> gemm = mkGemm();
    EXPECT_FALSE(sink(gemm)->setFusion(mkChain({mkAdd()}, {})));
}

TEST(FusionLowering, GemmAcceptsPerColumnBias)
{
    AgnosticChain ch = mkChain({mkAdd()}, {{"Add", perChannelOp(0)}});
    int sz[2] = {1, 4};
    ch.constBufs.push_back(Mat(2, sz, CV_32F, Scalar(0.5)));

    Ptr<Layer> gemm = mkGemm();
    EXPECT_TRUE(sink(gemm)->setFusion(ch));
}

TEST(FusionLowering, GemmRejectsPerRowBias)
{
    AgnosticChain ch = mkChain({mkAdd()}, {{"Add", perChannelOp(0)}});
    int sz[2] = {4, 1};
    ch.constBufs.push_back(Mat(2, sz, CV_32F, Scalar(0.5)));

    Ptr<Layer> gemm = mkGemm();
    EXPECT_FALSE(sink(gemm)->setFusion(ch));
}

TEST(FusionLowering, GemmRejectsMissingConstBuffer)
{
    Ptr<Layer> gemm = mkGemm();
    EXPECT_FALSE(sink(gemm)->setFusion(mkChain({mkAdd()}, {{"Add", perChannelOp(0)}})));
}

TEST(FusionLowering, GemmAcceptsBiasThenClip)
{
    AgnosticChain ch = mkChain({mkAdd(), mkClip()},
                                {{"Add", perChannelOp(0)}, {"Clip", clipOp(0.f, 6.f)}});
    int sz[2] = {1, 4};
    ch.constBufs.push_back(Mat(2, sz, CV_32F, Scalar(0.5)));

    Ptr<Layer> gemm = mkGemm();
    EXPECT_TRUE(sink(gemm)->setFusion(ch));
}

TEST(FusionLowering, SinkRefusesWhenAlreadyFused)
{
    LayerParams lp;
    Ptr<Layer> gemm = mkGemm();
    Ptr<Layer> gelu = GeluLayer::create(lp);

    ASSERT_TRUE(sink(gemm)->setFusion(mkChain({gelu}, {{"Gelu", FusionOperand()}})));
    EXPECT_FALSE(sink(gemm)->setFusion(mkChain({gelu}, {{"Gelu", FusionOperand()}})));
}

struct InterpOn
{
    bool was;
    InterpOn(bool on) : was(fusionInterpEnabled()) { setFusionInterpEnabled(on); }
    ~InterpOn() { setFusionInterpEnabled(was); }
};

TEST(FusionInterpTier, MulIsRefusedWhenInterpOff)
{
    InterpOn off(false);
    Ptr<Layer> gemm = mkGemm();
    EXPECT_FALSE(sink(gemm)->setFusion(mkChain({mkMul()}, {{"Mul", scalarOp(2.f)}})));
}

TEST(FusionInterpTier, MulIsAcceptedWhenInterpOn)
{
    InterpOn on(true);
    Ptr<Layer> gemm = mkGemm();
    EXPECT_TRUE(sink(gemm)->setFusion(mkChain({mkMul()}, {{"Mul", scalarOp(2.f)}})));
}

TEST(FusionInterpTier, MatMulAlsoGetsTheInterpTier)
{
    InterpOn on(true);
    Ptr<Layer> mm = mkMatMul();
    EXPECT_TRUE(sink(mm)->setFusion(mkChain({mkMul()}, {{"Mul", scalarOp(2.f)}})));
}

TEST(FusionInterpTier, InterpRefusesConstOnNonTrailingAxis)
{
    InterpOn on(true);
    AgnosticChain ch = mkChain({mkMul()}, {{"Mul", perChannelOp(0)}});
    int sz[2] = {4, 1};
    ch.constBufs.push_back(Mat(2, sz, CV_32F, Scalar(0.5)));

    std::vector<FusionStep> steps;
    EXPECT_FALSE(fusionLower(ch, steps));

    Ptr<Layer> gemm = mkGemm();
    EXPECT_FALSE(sink(gemm)->setFusion(ch));
}

TEST(FusionInterpTier, InterpCoversPrefixAndTierTwoCoversTail)
{
    InterpOn on(true);
    LayerParams lp;
    lp.set("negative_slope", 0.1f);
    Ptr<Layer> lrelu = ReLULayer::create(lp);

    AgnosticChain ch = mkChain({mkMul(), lrelu}, {{"Mul", scalarOp(2.f)}});
    ASSERT_EQ(ch.fgSteps, 1);
    ASSERT_EQ(ch.absorbed.size(), 2u);

    std::vector<FusionStep> steps;
    ASSERT_TRUE(fusionLower(ch, steps));
    ASSERT_EQ(steps.size(), 2u);
    EXPECT_TRUE(steps[0].interp);
    EXPECT_FALSE(steps[1].interp);
    EXPECT_NE(steps[1].fn, nullptr);
}

TEST(FusionInterpTier, InterpIsRefusedWhenTailIsOutsideTheIr)
{
    InterpOn on(true);
    AgnosticChain ch = mkChain({mkMul(), mkMul()}, {{"Mul", scalarOp(2.f)}});
    ASSERT_EQ(ch.fgSteps, 1);

    std::vector<FusionStep> steps;
    EXPECT_FALSE(fusionLower(ch, steps));
}

TEST(FusionInterpTier, InterpMatchesTheMathItReplaces)
{
    InterpOn on(true);
    AgnosticChain ch = mkChain({mkMul(), mkAdd()},
                                {{"Mul", scalarOp(3.f)}, {"Add", scalarOp(-1.f)}});

    std::vector<FusionStep> steps;
    ASSERT_TRUE(fusionLower(ch, steps));
    ASSERT_EQ(steps.size(), 1u);
    ASSERT_TRUE(steps[0].interp);

    Mat y(4, 5, CV_32F);
    RNG rng(0x51ee7);
    rng.fill(y, RNG::UNIFORM, -2.f, 2.f);
    Mat ref = y * 3.f - 1.f;

    fusionApply(steps, y);
    EXPECT_LE(cvtest::norm(y, ref, NORM_INF), 1e-6);
}

TEST(FusionInterpTier, InterpReadsPerChannelConstPerColumn)
{
    InterpOn on(true);
    AgnosticChain ch = mkChain({mkMul()}, {{"Mul", perChannelOp(0)}});
    int sz[2] = {1, 5};
    Mat k(2, sz, CV_32F);
    for (int j = 0; j < 5; j++)
        k.ptr<float>()[j] = 1.f + (float)j;
    ch.constBufs.push_back(k);

    std::vector<FusionStep> steps;
    ASSERT_TRUE(fusionLower(ch, steps));
    ASSERT_TRUE(steps[0].interp);

    Mat y(4, 5, CV_32F, Scalar(2.f));
    fusionApply(steps, y);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 5; j++)
            EXPECT_NEAR(y.at<float>(i, j), 2.f * (1.f + (float)j), 1e-6) << i << "," << j;
}

TEST(FusionInterpTier, InterpToggleRoundTrips)
{
    const bool was = fusionInterpEnabled();
    setFusionInterpEnabled(true);
    EXPECT_TRUE(fusionInterpEnabled());
    setFusionInterpEnabled(false);
    EXPECT_FALSE(fusionInterpEnabled());
    setFusionInterpEnabled(was);
}

TEST(FusionLowering, FusionToggleRoundTrips)
{
    const bool was = getAgnosticFusionEnabled();
    setAgnosticFusionEnabled(false);
    EXPECT_FALSE(getAgnosticFusionEnabled());
    setAgnosticFusionEnabled(true);
    EXPECT_TRUE(getAgnosticFusionEnabled());
    setAgnosticFusionEnabled(was);
}

#define FLOG "[ INFO     ] "

static std::string baseName(const std::string& p)
{
    const size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

static std::vector<cv::String> onnxModels()
{
    const std::string dir = cvtest::TS::ptr()->get_data_path() + "dnn/onnx/models/";
    std::vector<cv::String> files;
    try { cv::glob(dir + "*.onnx", files, false); } catch (const std::exception&) {}
    return files;
}

static std::string chainStr(const std::vector<std::string>& ch)
{
    std::string s;
    for (size_t k = 0; k < ch.size(); k++) s += (k ? " -> " : "") + ch[k];
    return s;
}

TEST(FusionRealONNX, CollectsChainsFromParsedGraphs)
{
    std::vector<cv::String> files = onnxModels();
    if (files.empty())
        throw SkipTestException("no .onnx models under testdata dnn/onnx/models/");

    const size_t limit = files.size();
    int loaded = 0, failed = 0, withChains = 0, totalChains = 0, longest = 0;
    int withAnchor = 0, anchorNoChain = 0;
    double parseMs = 0, collectMs = 0;
    std::map<std::string, int> absorbedOps, missedAfterAnchor;

    std::cout << FLOG << "scanning " << limit << " model(s)" << std::endl;

    for (size_t i = 0; i < limit; i++) {
        Net net;
        TickMeter tp;
        tp.start();
        try { net = readNetFromONNX(files[i], ENGINE_OPENCV); }
        catch (const std::exception&) { failed++; continue; }
        tp.stop();
        if (net.empty()) { failed++; continue; }
        loaded++;
        parseMs += tp.getTimeMilli();

        vector<std::string> types;
        graphOpTypes(net, types);
        bool hasAnchor = false;
        for (const std::string& t : types)
            if (classify(t) == OpShape::ANCHOR_TEMPLATE) { hasAnchor = true; break; }
        if (hasAnchor) withAnchor++;

        vector<vector<std::string> > chains;
        TickMeter tc;
        tc.start();
        collectAgnosticChainTypes(net, chains);
        tc.stop();
        collectMs += tc.getTimeMilli();

        if (hasAnchor && chains.empty()) {
            anchorNoChain++;
            for (const std::string& t : types)
                if (classify(t) != OpShape::ANCHOR_TEMPLATE) missedAfterAnchor[t]++;
            if (anchorNoChain <= 10)
                std::cout << FLOG << "anchor but no chain: " << baseName(files[i])
                          << " [" << chainStr(types) << "]" << std::endl;
        }

        for (const vector<std::string>& ch : chains) {
            ASSERT_GE(ch.size(), 2u) << baseName(files[i]);
            EXPECT_EQ(classify(ch[0]), OpShape::ANCHOR_TEMPLATE) << "anchor " << ch[0];
            for (size_t k = 1; k < ch.size(); k++) {
                EXPECT_NE(classify(ch[k]), OpShape::ANCHOR_TEMPLATE) << "absorbed " << ch[k];
                EXPECT_NE(classify(ch[k]), OpShape::ANCHOR_REDUCE) << "absorbed " << ch[k];
                absorbedOps[ch[k]]++;
            }
            longest = std::max(longest, (int)ch.size());
        }

        if (!chains.empty()) {
            withChains++;
            totalChains += (int)chains.size();
            std::cout << FLOG << baseName(files[i]) << ": " << chains.size()
                      << " chain(s), collect " << tc.getTimeMilli() << " ms" << std::endl;
            for (size_t k = 0; k < chains.size() && k < 8; k++)
                std::cout << FLOG << "    " << chainStr(chains[k]) << std::endl;
            if (chains.size() > 8)
                std::cout << FLOG << "    ... " << (chains.size() - 8) << " more" << std::endl;
        }
    }

    std::cout << FLOG << "---- summary ----" << std::endl;
    std::cout << FLOG << "parsed ok         : " << loaded << " (failed " << failed << ")" << std::endl;
    std::cout << FLOG << "with anchor op    : " << withAnchor << std::endl;
    std::cout << FLOG << "anchor, no chain  : " << anchorNoChain << std::endl;
    std::cout << FLOG << "models with chains: " << withChains << std::endl;
    std::cout << FLOG << "total chains      : " << totalChains << std::endl;
    std::cout << FLOG << "longest chain     : " << longest << " nodes" << std::endl;
    std::cout << FLOG << "parse time        : " << parseMs << " ms" << std::endl;
    std::cout << FLOG << "collect time      : " << collectMs << " ms" << std::endl;
    for (std::map<std::string, int>::const_iterator it = absorbedOps.begin(); it != absorbedOps.end(); ++it)
        std::cout << FLOG << "absorbed " << it->first << " x" << it->second << std::endl;
    std::cout << FLOG << "-- ops seen in anchor-but-no-chain models --" << std::endl;
    for (std::map<std::string, int>::const_iterator it = missedAfterAnchor.begin(); it != missedAfterAnchor.end(); ++it)
        std::cout << FLOG << "  " << it->first << " x" << it->second
                  << (classify(it->first) == OpShape::MAP ? "  (MAP)" : "") << std::endl;

    EXPECT_GT(loaded, 0);
}

TEST(FusionRealONNX, FusedOutputMatchesUnfusedBitExact)
{
    std::vector<cv::String> files = onnxModels();
    if (files.empty())
        throw SkipTestException("no .onnx models under testdata dnn/onnx/models/");

    const bool was = getAgnosticFusionEnabled();
    int compared = 0, inexact = 0;
    std::vector<std::pair<double, std::string> > off;
    std::vector<std::string> flaky;

    for (size_t i = 0; i < files.size(); i++) {
        setAgnosticFusionEnabled(true);
        Net fused;
        try { fused = readNetFromONNX(files[i], ENGINE_OPENCV); }
        catch (const std::exception&) { continue; }
        if (fused.empty()) continue;

        setAgnosticFusionEnabled(false);
        Net plain;
        try { plain = readNetFromONNX(files[i], ENGINE_OPENCV); }
        catch (const std::exception&) { continue; }
        if (plain.empty()) continue;

        MatShape ishape;
        if (!graphInputShape(fused, ishape))
            continue;

        Mat inp, outA, outB, outC, outD;
        try {
            inp.create((int)ishape.size(), ishape.data(), CV_32F);
            if (inp.empty()) continue;
            RNG rng(0x1234abcd);
            rng.fill(inp, RNG::UNIFORM, -1.f, 1.f);

            fused.setInput(inp);
            outA = fused.forward().clone();
            plain.setInput(inp);
            outB = plain.forward().clone();
            fused.setInput(inp);
            outC = fused.forward().clone();
            plain.setInput(inp);
            outD = plain.forward().clone();
        }
        catch (const std::exception&) { continue; }

        if (outA.empty() || outB.empty() || outA.size != outB.size || outA.type() != outB.type())
            continue;

        compared++;
        if (std::memcmp(outA.data, outB.data, outA.total() * outA.elemSize()) != 0) {
            inexact++;
            const double nrm = cvtest::norm(outA, outB, NORM_INF);
            const double mag = std::max(cvtest::norm(outB, NORM_INF), 1e-6);
            const double l2  = cvtest::norm(outA, outB, NORM_L2);
            const double m2  = std::max(cvtest::norm(outB, NORM_L2), 1e-6);
            const bool detA = outC.size == outA.size && outC.type() == outA.type() &&
                              std::memcmp(outA.data, outC.data, outA.total() * outA.elemSize()) == 0;
            const bool detB = outD.size == outB.size && outD.type() == outB.type() &&
                              std::memcmp(outB.data, outD.data, outB.total() * outB.elemSize()) == 0;
            const double selfA = detA ? 0.0 : cvtest::norm(outA, outC, NORM_INF) / mag;
            const double selfB = detB ? 0.0 : cvtest::norm(outB, outD, NORM_INF) / mag;
            const bool det = detA && detB;

            std::ostringstream s;
            s << "rel=" << nrm / mag << "  linf=" << nrm << "  ref=" << mag
              << "  relL2=" << l2 / m2 << "  selfFused=" << selfA << "  selfPlain=" << selfB
              << "  n=" << outB.total() << "  " << baseName(files[i]);
            if (det)
                off.push_back(std::make_pair(nrm / mag, s.str()));
            else
                flaky.push_back(s.str());
        }
    }

    setAgnosticFusionEnabled(was);
    std::sort(off.begin(), off.end());
    std::cout << FLOG << "compared " << compared << " model(s), "
              << (compared - inexact) << " bit-exact, " << inexact << " inexact, "
              << flaky.size() << " nondeterministic" << std::endl;
    for (size_t k = off.size(); k > 0; k--)
        std::cout << FLOG << "  " << off[k - 1].second << std::endl;
    for (size_t k = 0; k < flaky.size(); k++)
        std::cout << FLOG << "  NONDET " << flaky[k] << std::endl;
    for (size_t k = 0; k < off.size(); k++)
        EXPECT_LE(off[k].first, 1e-3) << off[k].second;
}

static bool opHist(const cv::String& f, bool fuse, std::map<std::string, int>& h)
{
    setAgnosticFusionEnabled(fuse);
    Net n;
    try { n = readNetFromONNX(f, ENGINE_OPENCV); }
    catch (const std::exception&) { return false; }
    if (n.empty()) return false;

    MatShape sh;
    if (!graphInputShape(n, sh)) return false;

    try {
        Mat in;
        in.create((int)sh.size(), sh.data(), CV_32F);
        if (in.empty()) return false;
        RNG rng(0x1234abcd);
        rng.fill(in, RNG::UNIFORM, -1.f, 1.f);
        n.setInput(in);
        n.forward();
    }
    catch (const std::exception&) { return false; }

    vector<std::string> t;
    graphOpTypes(n, t);
    if (t.empty()) return false;

    h.clear();
    for (const std::string& s : t) h[s]++;
    return true;
}

TEST(FusionRealONNX, FusedGraphIsNeverLarger)
{
    std::vector<cv::String> files = onnxModels();
    if (files.empty())
        throw SkipTestException("no .onnx models under testdata dnn/onnx/models/");

    const bool was = getAgnosticFusionEnabled();
    int cmp = 0, win = 0;
    std::vector<std::string> grew;

    for (size_t i = 0; i < files.size(); i++) {
        std::map<std::string, int> hn, ho;
        if (!opHist(files[i], true, hn)) continue;
        if (!opHist(files[i], false, ho)) continue;

        cmp++;
        int nn = 0, no = 0;
        for (const auto& p : hn) nn += p.second;
        for (const auto& p : ho) no += p.second;
        if (nn < no) win++;

        for (const auto& p : hn) {
            auto o = ho.find(p.first);
            const int c = o == ho.end() ? 0 : o->second;
            if (p.second > c)
                grew.push_back(baseName(files[i]) + " " + p.first +
                               " fused=" + std::to_string(p.second) + " plain=" + std::to_string(c));
        }
    }

    setAgnosticFusionEnabled(was);
    std::cout << FLOG << "compared " << cmp << " model(s), " << win
              << " with a smaller graph, " << grew.size() << " with more of some op" << std::endl;
    for (size_t k = 0; k < grew.size(); k++)
        std::cout << FLOG << "  GREW " << grew[k] << std::endl;
    EXPECT_GT(cmp, 0);
    EXPECT_TRUE(grew.empty());
}

TEST(FusionRealONNX, CollectIsRepeatableAndNonMutating)
{
    std::vector<cv::String> files = onnxModels();
    if (files.empty())
        throw SkipTestException("no .onnx models under testdata dnn/onnx/models/");

    int checked = 0;
    for (size_t i = 0; i < files.size() && checked < 10; i++) {
        Net net;
        try { net = readNetFromONNX(files[i], ENGINE_OPENCV); }
        catch (const std::exception&) { continue; }
        if (net.empty()) continue;

        vector<vector<std::string> > first, second;
        collectAgnosticChainTypes(net, first);
        collectAgnosticChainTypes(net, second);

        EXPECT_EQ(first, second) << baseName(files[i]);
        checked++;
    }

    std::cout << FLOG << "repeat-checked " << checked << " model(s)" << std::endl;
    EXPECT_GT(checked, 0);
}

}} // namespace
