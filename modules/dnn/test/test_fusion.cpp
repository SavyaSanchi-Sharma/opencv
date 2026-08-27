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
    Ptr<FusionGraph> g = graphFromRecipe(r);
    CV_Assert(g);
    return evalFusionGraph(*g, x, kNoBufs, 0);
}

TEST(Fusion, IdenticalRecipesCollapseToTheSameNode)
{
    FusionGraphBuilder arena;
    const int in = arena.push(FusionEltwiseOp::INPUT, {});
    FusionRecipe r;
    reluRecipe(r);

    EXPECT_EQ(appendFusionOp(arena, in, r), appendFusionOp(arena, in, r));
    EXPECT_EQ(appendFusionOp(arena, in, r), appendFusionOp(arena, in, r));
    EXPECT_EQ(3u, arena.size());

    FusionGraphBuilder b;
    const int i2 = b.push(FusionEltwiseOp::INPUT, {});
    const int k = b.push(FusionEltwiseOp::CONST, {}, 7.f);
    EXPECT_EQ(b.push(FusionEltwiseOp::ADD, {i2, k}), b.push(FusionEltwiseOp::ADD, {k, i2}));
    EXPECT_NE(b.push(FusionEltwiseOp::SUB, {i2, k}), b.push(FusionEltwiseOp::SUB, {k, i2}));
}

TEST(Fusion, ConeIsBoundedIndependentlyOfArenaSize)
{
    FusionGraphBuilder arena;
    const int in = arena.push(FusionEltwiseOp::INPUT, {});
    FusionRecipe a, b;
    reluRecipe(a);
    geluRecipe(b);
    const int rootA = appendFusionOp(arena, in, a);
    const int rootB = appendFusionOp(arena, in, b);
    ASSERT_GE(rootA, 0);
    ASSERT_GE(rootB, 0);

    std::vector<char> scratch;
    EXPECT_EQ(3, coneSize(arena.graph(), rootA, scratch));
    EXPECT_EQ(9, coneSize(arena.graph(), rootB, scratch));
    EXPECT_GT(arena.size(), (size_t)9);
}

TEST(Fusion, ExtractionYieldsAStandaloneGraph)
{
    FusionGraphBuilder arena;
    const int in = arena.push(FusionEltwiseOp::INPUT, {});
    FusionRecipe a, b;
    reluRecipe(a);
    geluRecipe(b);
    appendFusionOp(arena, in, a);
    const int rootB = appendFusionOp(arena, in, b);

    Ptr<FusionGraph> g = extractSubgraph(arena.graph(), rootB, std::vector<Mat>());
    ASSERT_TRUE(g);
    EXPECT_EQ(9u, g->size());
    EXPECT_EQ(FusionEltwiseOp::INPUT, g->nodes()[0].op);
    EXPECT_EQ((int)g->size() - 1, g->outputNode);
    EXPECT_NEAR(0.5f * 1.5f * (1.f + std::erf(1.5f * (float)M_SQRT1_2)),
                evalFusionGraph(*g, 1.5f, kNoBufs, 0), 1e-5);

    EXPECT_FALSE(extractSubgraph(arena.graph(), -1, std::vector<Mat>()));
    EXPECT_FALSE(extractSubgraph(arena.graph(), (int)arena.size(), std::vector<Mat>()));
}

TEST(Fusion, OverLimitConeIsRefusedNotEvaluated)
{
    FusionGraphBuilder arena;
    int cur = arena.push(FusionEltwiseOp::INPUT, {});
    std::vector<char> scratch;
    int steps = 0;
    while (coneSize(arena.graph(), cur, scratch) <= FUSION_MAX_NODES && steps < 200) {
        FusionRecipe r;
        geluRecipe(r);
        const int next = appendFusionOp(arena, cur, r);
        ASSERT_GE(next, 0);
        cur = next;
        steps++;
    }
    ASSERT_GT(coneSize(arena.graph(), cur, scratch), FUSION_MAX_NODES);
    EXPECT_FALSE(extractSubgraph(arena.graph(), cur, std::vector<Mat>()));
}

TEST(Fusion, RecipesMatchClosedForm)
{
    const float xs[] = { -3.f, -0.5f, 0.f, 0.25f, 1.f, 4.f };
    FusionRecipe r;
    for (float x : xs) {
        r = FusionRecipe(); reluRecipe(r);
        EXPECT_FLOAT_EQ(std::max(x, 0.f), eval1(r, x)) << "relu " << x;

        r = FusionRecipe(); clampRecipe(r, 0.f, 6.f);
        EXPECT_FLOAT_EQ(std::min(std::max(x, 0.f), 6.f), eval1(r, x)) << "clip " << x;

        r = FusionRecipe(); sigmoidRecipe(r);
        EXPECT_NEAR(1.f / (1.f + std::exp(-x)), eval1(r, x), 1e-5) << "sigmoid " << x;

        r = FusionRecipe(); geluRecipe(r);
        EXPECT_NEAR(0.5f * x * (1.f + std::erf(x * (float)M_SQRT1_2)), eval1(r, x), 1e-5)
            << "gelu " << x;

        r = FusionRecipe(); unaryRecipe(r, FusionEltwiseOp::TANH);
        EXPECT_NEAR(std::tanh(x), eval1(r, x), 1e-6) << "tanh " << x;

        r = FusionRecipe(); expRecipe(r, 2.f, 5.f);
        EXPECT_NEAR(std::exp(2.f * x + 5.f), eval1(r, x), 1e-2) << "scaled exp " << x;
    }
    r = FusionRecipe(); expRecipe(r, 1.f, 0.f);
    EXPECT_EQ(1, r.n);
    r = FusionRecipe(); expRecipe(r, 2.f, 5.f);
    EXPECT_EQ(5, r.n);
}

TEST(Fusion, WellFormedCatchesWhatArityChecksMiss)
{
    FusionRecipe unset;
    unset.node[0].op = FusionEltwiseOp::PER_CHANNEL_CONST;
    unset.n = 1;
    EXPECT_FALSE(unset.wellFormed());

    FusionRecipe explicitInput;
    explicitInput.node[0].op = FusionEltwiseOp::INPUT;
    explicitInput.n = 1;
    EXPECT_FALSE(explicitInput.wellFormed());

    FusionRecipe selfRef;
    selfRef.node[0].op = FusionEltwiseOp::MAX;
    selfRef.node[0].a = -1;
    selfRef.node[0].b = 0;
    selfRef.n = 1;
    EXPECT_FALSE(selfRef.wellFormed());

    FusionGraphBuilder arena;
    const int in = arena.push(FusionEltwiseOp::INPUT, {});
    EXPECT_EQ(-1, appendFusionOp(arena, in, unset));
    EXPECT_EQ(1u, arena.size());
}

TEST(Fusion, ReversedSubIsRefused)
{
    FusionRecipe r;
    ValueSource vs;
    vs.hasFoldedOperand = true;
    vs.scalar = 3.f;
    vs.flowIsFirst = false;
    EXPECT_FALSE(binaryConstRecipe(r, FusionEltwiseOp::SUB, vs));

    vs.flowIsFirst = true;
    r = FusionRecipe();
    ASSERT_TRUE(binaryConstRecipe(r, FusionEltwiseOp::SUB, vs));
    EXPECT_FLOAT_EQ(-1.f, eval1(r, 2.f));
}

TEST(Fusion, ActivationMatchRecognizesAndRefuses)
{
    int activ = ACTIV_NONE;
    std::vector<float> params;
    FusionRecipe r;

    r = FusionRecipe(); clampRecipe(r, 0.f, 6.f);
    ASSERT_TRUE(matchFusionActivation(*graphFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_CLIP, activ);
    ASSERT_EQ(2u, params.size());
    EXPECT_FLOAT_EQ(0.f, params[0]);
    EXPECT_FLOAT_EQ(6.f, params[1]);

    r = FusionRecipe(); reluRecipe(r);
    ASSERT_TRUE(matchFusionActivation(*graphFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_RELU, activ);

    r = FusionRecipe(); sigmoidRecipe(r);
    ASSERT_TRUE(matchFusionActivation(*graphFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_SIGMOID, activ);

    r = FusionRecipe(); geluRecipe(r);
    ASSERT_TRUE(matchFusionActivation(*graphFromRecipe(r), activ, params));
    EXPECT_EQ(ACTIV_GELU, activ);

    r = FusionRecipe(); unaryRecipe(r, FusionEltwiseOp::SQRT);
    EXPECT_FALSE(matchFusionActivation(*graphFromRecipe(r), activ, params));

    FusionGraphBuilder arena;
    arena.push(FusionEltwiseOp::INPUT, {});
    r = FusionRecipe(); reluRecipe(r);
    appendFusionOp(arena, 0, r);
    EXPECT_FALSE(matchFusionActivation(*arena.release(), activ, params));
}

TEST(Fusion, ApplyTakesKernelPathThenInterpreterPath)
{
    FusionRecipe r;
    clampRecipe(r, 0.f, 6.f);
    FusionApply kern;
    ASSERT_TRUE(prepareFusionApply(graphFromRecipe(r), kern));
    ASSERT_TRUE(kern.fn != nullptr);

    int n = 5;
    Mat y(1, &n, CV_32F);
    const float src[] = { -2.f, 0.f, 3.f, 6.f, 9.f };
    std::copy(src, src + n, y.ptr<float>());
    applyFusion(kern, y);
    const float want[] = { 0.f, 0.f, 3.f, 6.f, 6.f };
    for (int i = 0; i < n; i++)
        EXPECT_FLOAT_EQ(want[i], y.ptr<float>()[i]) << "clip i=" << i;

    r = FusionRecipe();
    unaryRecipe(r, FusionEltwiseOp::SQRT);
    FusionApply interp;
    ASSERT_TRUE(prepareFusionApply(graphFromRecipe(r), interp));
    EXPECT_TRUE(interp.fn == nullptr);

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
    const int in = arena.push(FusionEltwiseOp::INPUT, {});
    FusionRecipe r;
    ValueSource vs;
    vs.hasFoldedOperand = true;
    vs.bufferId = 1;
    ASSERT_TRUE(binaryConstRecipe(r, FusionEltwiseOp::MUL, vs));
    const int root = appendFusionOp(arena, in, r);
    ASSERT_GE(root, 0);

    int one = 1, three = 3;
    Mat b0(1, &one, CV_32F);
    b0.ptr<float>()[0] = 1.f;
    Mat b1(1, &three, CV_32F);
    b1.ptr<float>()[0] = 2.f;
    b1.ptr<float>()[1] = 3.f;
    b1.ptr<float>()[2] = 4.f;

    std::vector<Mat> tooFew(1, b0);
    EXPECT_FALSE(extractSubgraph(arena.graph(), root, tooFew));

    std::vector<Mat> bufs;
    bufs.push_back(b0);
    bufs.push_back(b1);
    Ptr<FusionGraph> expr = extractSubgraph(arena.graph(), root, bufs);
    ASSERT_TRUE(expr);
    bool seen = false;
    for (const FusionNode& nd : expr->nodes()) {
        if (nd.op == FusionEltwiseOp::PER_CHANNEL_CONST) {
            EXPECT_EQ(1, nd.constBufferId);
            seen = true;
        }
    }
    EXPECT_TRUE(seen);

    FusionApply fa;
    ASSERT_TRUE(prepareFusionApply(expr, fa));
    EXPECT_TRUE(fa.fn == nullptr);

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
    const int in = arena.push(FusionEltwiseOp::INPUT, {});

    FusionRecipe r;
    ValueSource vs;
    vs.hasFoldedOperand = true;
    vs.bufferId = 0;
    ASSERT_TRUE(binaryConstRecipe(r, FusionEltwiseOp::MUL, vs));

    const int rootA = appendFusionOp(arena, in, r);
    const int rootB = appendFusionOp(arena, in, r);
    ASSERT_GE(rootA, 0);
    EXPECT_EQ(rootA, rootB);

    int three = 3;
    Mat ba(1, &three, CV_32F), bb(1, &three, CV_32F);
    for (int i = 0; i < 3; i++) { ba.ptr<float>()[i] = 2.f; bb.ptr<float>()[i] = 10.f; }

    Ptr<FusionGraph> ea = extractSubgraph(arena.graph(), rootA, std::vector<Mat>(1, ba));
    Ptr<FusionGraph> eb = extractSubgraph(arena.graph(), rootB, std::vector<Mat>(1, bb));
    ASSERT_TRUE(ea);
    ASSERT_TRUE(eb);

    ASSERT_EQ(1u, ea->constBufs.size());
    ASSERT_EQ(1u, eb->constBufs.size());
    EXPECT_FLOAT_EQ(2.f,  ea->constBufs[0].ptr<float>()[0]);
    EXPECT_FLOAT_EQ(10.f, eb->constBufs[0].ptr<float>()[0]);

    FusionApply fa, fb;
    ASSERT_TRUE(prepareFusionApply(ea, fa));
    ASSERT_TRUE(prepareFusionApply(eb, fb));

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
