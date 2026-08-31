// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/fusion_graph.hpp"
#include "../src/layers/cpu_kernels/fusion_apply.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;

static const std::vector<const float*> kNoBufs;

static float eval1(const FusionRecipe& r, float x)
{
    Ptr<FusionGraph> g = patternFromRecipe(r);
    CV_Assert(g);
    return evalFusionGraph(*g, x, kNoBufs, 0);
}

TEST(Fusion, IdenticalRecipesCollapseToTheSameNode)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    FusionRecipe r;
    const int zero = r.constant(0.f);
    r.binary(FusionEltwiseOp::MAX, FusionRecipe::INPUT_VALUE, zero);

    EXPECT_EQ(instantiateRecipe(arena, in, r), instantiateRecipe(arena, in, r));
    EXPECT_EQ(instantiateRecipe(arena, in, r), instantiateRecipe(arena, in, r));
    EXPECT_EQ(3u, arena.size());

    FusionGraphBuilder b;
    const int i2 = b.internNode(FusionEltwiseOp::INPUT, {});
    const int k = b.internNode(FusionEltwiseOp::CONST, {}, 7.f);
    EXPECT_EQ(b.internNode(FusionEltwiseOp::ADD, {i2, k}), b.internNode(FusionEltwiseOp::ADD, {k, i2}));
    EXPECT_NE(b.internNode(FusionEltwiseOp::SUB, {i2, k}), b.internNode(FusionEltwiseOp::SUB, {k, i2}));
}

TEST(Fusion, ConeIsBoundedIndependentlyOfArenaSize)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    FusionRecipe a, b;
    const int zero = a.constant(0.f);
    a.binary(FusionEltwiseOp::MAX, FusionRecipe::INPUT_VALUE, zero);
    geluRecipe(b);
    const int rootA = instantiateRecipe(arena, in, a);
    const int rootB = instantiateRecipe(arena, in, b);
    ASSERT_GE(rootA, 0);
    ASSERT_GE(rootB, 0);

    std::vector<char> scratch;
    EXPECT_EQ(3, reachableNodeCount(arena.graph(), rootA, scratch));
    EXPECT_EQ(9, reachableNodeCount(arena.graph(), rootB, scratch));
    EXPECT_GT(arena.size(), (size_t)9);
}

TEST(Fusion, ExtractionYieldsAStandaloneGraph)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    FusionRecipe a, b;
    const int zero = a.constant(0.f);
    a.binary(FusionEltwiseOp::MAX, FusionRecipe::INPUT_VALUE, zero);
    geluRecipe(b);
    instantiateRecipe(arena, in, a);
    const int rootB = instantiateRecipe(arena, in, b);

    Ptr<FusionGraph> g = extractExpression(arena.graph(), rootB, std::vector<Mat>());
    ASSERT_TRUE(g);
    EXPECT_EQ(9u, g->size());
    EXPECT_EQ(FusionEltwiseOp::INPUT, g->nodes()[0].op);
    EXPECT_EQ((int)g->size() - 1, g->outputNode);
    EXPECT_NEAR(0.5f * 1.5f * (1.f + std::erf(1.5f * (float)M_SQRT1_2)),
                evalFusionGraph(*g, 1.5f, kNoBufs, 0), 1e-5);

    EXPECT_FALSE(extractExpression(arena.graph(), -1, std::vector<Mat>()));
    EXPECT_FALSE(extractExpression(arena.graph(), (int)arena.size(), std::vector<Mat>()));
}

TEST(Fusion, OverLimitConeIsRefusedNotEvaluated)
{
    FusionGraphBuilder arena;
    int cur = arena.internNode(FusionEltwiseOp::INPUT, {});
    std::vector<char> scratch;
    int steps = 0;
    while (reachableNodeCount(arena.graph(), cur, scratch) <= FUSION_MAX_EXPR_NODES && steps < 200) {
        FusionRecipe r;
        geluRecipe(r);
        const int next = instantiateRecipe(arena, cur, r);
        ASSERT_GE(next, 0);
        cur = next;
        steps++;
    }
    ASSERT_GT(reachableNodeCount(arena.graph(), cur, scratch), FUSION_MAX_EXPR_NODES);
    EXPECT_FALSE(extractExpression(arena.graph(), cur, std::vector<Mat>()));
}

TEST(Fusion, RecipesMatchClosedForm)
{
    const float xs[] = { -3.f, -0.5f, 0.f, 0.25f, 1.f, 4.f };
    FusionRecipe r;
    for (float x : xs) {
        r = FusionRecipe();
        r.binary(FusionEltwiseOp::MAX, FusionRecipe::INPUT_VALUE, r.constant(0.f));
        EXPECT_FLOAT_EQ(std::max(x, 0.f), eval1(r, x)) << "relu " << x;

        r = FusionRecipe();
        r.clamp(FusionRecipe::INPUT_VALUE, 0.f, 6.f);
        EXPECT_FLOAT_EQ(std::min(std::max(x, 0.f), 6.f), eval1(r, x)) << "clip " << x;

        r = FusionRecipe(); sigmoidRecipe(r);
        EXPECT_NEAR(1.f / (1.f + std::exp(-x)), eval1(r, x), 1e-5) << "sigmoid " << x;

        r = FusionRecipe(); geluRecipe(r);
        EXPECT_NEAR(0.5f * x * (1.f + std::erf(x * (float)M_SQRT1_2)), eval1(r, x), 1e-5)
            << "gelu " << x;

        r = FusionRecipe();
        r.unary(FusionEltwiseOp::TANH, FusionRecipe::INPUT_VALUE);
        EXPECT_NEAR(std::tanh(x), eval1(r, x), 1e-6) << "tanh " << x;

        r = FusionRecipe();
        const int scaled = r.binary(FusionEltwiseOp::MUL, FusionRecipe::INPUT_VALUE, r.constant(2.f));
        r.unary(FusionEltwiseOp::EXP, r.binary(FusionEltwiseOp::ADD, scaled, r.constant(5.f)));
        EXPECT_NEAR(std::exp(2.f * x + 5.f), eval1(r, x), 1e-2) << "scaled exp " << x;
    }
}

TEST(Fusion, EmptyRecipeIsRefused)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    EXPECT_EQ(-1, instantiateRecipe(arena, in, FusionRecipe()));
    EXPECT_EQ(1u, arena.size());
}

TEST(Fusion, ReversedSubIsRefused)
{
    LayerParams lp;
    lp.set("operation", "sub");
    Ptr<Layer> sub = NaryEltwiseLayer::create(lp);
    ASSERT_TRUE(sub);

    FusionRecipe r;
    ConstOperand vs;
    vs.hasValue = true;
    vs.value = 3.f;
    vs.flowIsFirstInput = false;
    EXPECT_FALSE(sub->describeMath(r, vs));

    vs.flowIsFirstInput = true;
    r = FusionRecipe();
    ASSERT_TRUE(sub->describeMath(r, vs));
    EXPECT_FLOAT_EQ(-1.f, eval1(r, 2.f));
}

TEST(Fusion, ActivationMatchRecognizesAndRefuses)
{
    int activ = ACTIV_NONE;
    std::vector<float> params;
    FusionRecipe r;

    r = FusionRecipe();
    r.clamp(FusionRecipe::INPUT_VALUE, 0.f, 6.f);
    ASSERT_TRUE(matchKnownActivation(*patternFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_CLIP, activ);
    ASSERT_EQ(2u, params.size());
    EXPECT_FLOAT_EQ(0.f, params[0]);
    EXPECT_FLOAT_EQ(6.f, params[1]);

    r = FusionRecipe();
    r.binary(FusionEltwiseOp::MAX, FusionRecipe::INPUT_VALUE, r.constant(0.f));
    ASSERT_TRUE(matchKnownActivation(*patternFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_RELU, activ);

    r = FusionRecipe(); sigmoidRecipe(r);
    ASSERT_TRUE(matchKnownActivation(*patternFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_SIGMOID, activ);

    r = FusionRecipe(); geluRecipe(r);
    ASSERT_TRUE(matchKnownActivation(*patternFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_GELU, activ);

    r = FusionRecipe();
    r.unary(FusionEltwiseOp::SQRT, FusionRecipe::INPUT_VALUE);
    EXPECT_FALSE(matchKnownActivation(*patternFromRecipe(r), activ, params));

    FusionGraphBuilder arena;
    arena.internNode(FusionEltwiseOp::INPUT, {});
    r = FusionRecipe();
    r.binary(FusionEltwiseOp::MAX, FusionRecipe::INPUT_VALUE, r.constant(0.f));
    instantiateRecipe(arena, 0, r);
    EXPECT_FALSE(matchKnownActivation(*arena.sharedGraph(), activ, params));
}

TEST(Fusion, ApplyTakesKernelPathThenInterpreterPath)
{
    FusionRecipe r;
    r.clamp(FusionRecipe::INPUT_VALUE, 0.f, 6.f);
    PreparedFusion kern;
    ASSERT_TRUE(prepareFusion(patternFromRecipe(r), kern));
    ASSERT_TRUE(kern.activationFn != nullptr);

    int n = 5;
    Mat y(1, &n, CV_32F);
    const float src[] = { -2.f, 0.f, 3.f, 6.f, 9.f };
    std::copy(src, src + n, y.ptr<float>());
    applyFusion(kern, y);
    const float want[] = { 0.f, 0.f, 3.f, 6.f, 6.f };
    for (int i = 0; i < n; i++)
        EXPECT_FLOAT_EQ(want[i], y.ptr<float>()[i]) << "clip i=" << i;

    r = FusionRecipe();
    r.unary(FusionEltwiseOp::SQRT, FusionRecipe::INPUT_VALUE);
    PreparedFusion interp;
    ASSERT_TRUE(prepareFusion(patternFromRecipe(r), interp));
    EXPECT_TRUE(interp.activationFn == nullptr);

    int big = (1 << 16) + 17;
    Mat z(1, &big, CV_32F);
    for (int i = 0; i < big; i++)
        z.ptr<float>()[i] = (float)(i % 100);
    applyFusion(interp, z);
    for (int i = 0; i < big; i += 997)
        EXPECT_NEAR(std::sqrt((float)(i % 100)), z.ptr<float>()[i], 1e-5) << "sqrt i=" << i;
}

TEST(Fusion, PerChannelConstIndexesTheLastAxis)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    FusionRecipe r;
    r.binary(FusionEltwiseOp::MUL, FusionRecipe::INPUT_VALUE, r.perChannelConstant(1));
    const int root = instantiateRecipe(arena, in, r);
    ASSERT_GE(root, 0);

    int one = 1, three = 3;
    Mat b0(1, &one, CV_32F);
    b0.ptr<float>()[0] = 1.f;
    Mat b1(1, &three, CV_32F);
    b1.ptr<float>()[0] = 2.f;
    b1.ptr<float>()[1] = 3.f;
    b1.ptr<float>()[2] = 4.f;

    std::vector<Mat> tooFew(1, b0);
    EXPECT_FALSE(extractExpression(arena.graph(), root, tooFew));

    std::vector<Mat> bufs;
    bufs.push_back(b0);
    bufs.push_back(b1);
    Ptr<FusionGraph> expr = extractExpression(arena.graph(), root, bufs);
    ASSERT_TRUE(expr);
    bool seen = false;
    for (const FusionNode& nd : expr->nodes()) {
        if (nd.op == FusionEltwiseOp::PER_CHANNEL_CONST) {
            EXPECT_EQ(1, nd.constBufferId);
            seen = true;
        }
    }
    EXPECT_TRUE(seen);

    PreparedFusion fa;
    ASSERT_TRUE(prepareFusion(expr, fa));
    EXPECT_TRUE(fa.activationFn == nullptr);

    int sz[] = { 2, 3 };
    Mat y(2, sz, CV_32F);
    for (int i = 0; i < 6; i++)
        y.ptr<float>()[i] = 1.f;
    applyFusion(fa, y);
    const float want[] = { 2.f, 3.f, 4.f, 2.f, 3.f, 4.f };
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(want[i], y.ptr<float>()[i]) << "i=" << i;
}

TEST(Fusion, SharedRootKeepsEachChainsOwnBuffers)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});

    FusionRecipe r;
    r.binary(FusionEltwiseOp::MUL, FusionRecipe::INPUT_VALUE, r.perChannelConstant(0));

    const int rootA = instantiateRecipe(arena, in, r);
    const int rootB = instantiateRecipe(arena, in, r);
    ASSERT_GE(rootA, 0);
    EXPECT_EQ(rootA, rootB);

    int three = 3;
    Mat ba(1, &three, CV_32F), bb(1, &three, CV_32F);
    for (int i = 0; i < 3; i++) { ba.ptr<float>()[i] = 2.f; bb.ptr<float>()[i] = 10.f; }

    Ptr<FusionGraph> ea = extractExpression(arena.graph(), rootA, std::vector<Mat>(1, ba));
    Ptr<FusionGraph> eb = extractExpression(arena.graph(), rootB, std::vector<Mat>(1, bb));
    ASSERT_TRUE(ea);
    ASSERT_TRUE(eb);

    ASSERT_EQ(1u, ea->constBufs.size());
    ASSERT_EQ(1u, eb->constBufs.size());
    EXPECT_FLOAT_EQ(2.f,  ea->constBufs[0].ptr<float>()[0]);
    EXPECT_FLOAT_EQ(10.f, eb->constBufs[0].ptr<float>()[0]);

    PreparedFusion fa, fb;
    ASSERT_TRUE(prepareFusion(ea, fa));
    ASSERT_TRUE(prepareFusion(eb, fb));

    int sz[] = { 1, 3 };
    Mat ya(2, sz, CV_32F), yb(2, sz, CV_32F);
    for (int i = 0; i < 3; i++) { ya.ptr<float>()[i] = 1.f; yb.ptr<float>()[i] = 1.f; }
    applyFusion(fa, ya);
    applyFusion(fb, yb);
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(2.f,  ya.ptr<float>()[i]) << "A i=" << i;
        EXPECT_FLOAT_EQ(10.f, yb.ptr<float>()[i]) << "B i=" << i;
    }
}

}} // namespace opencv_test
