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

static float eval1(const LayerMath& r, float x)
{
    Ptr<FusionGraph> g = fusion::fromMath(r);
    CV_Assert(g);
    return fusion::eval(*g, x, kNoBufs, 0);
}

TEST(Fusion, IdenticalMathCollapsesToTheSameNode)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath r;
    const int zero = r.constant(0.f);
    r.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, zero);

    EXPECT_EQ(fusion::instantiate(arena, in, r), fusion::instantiate(arena, in, r));
    EXPECT_EQ(fusion::instantiate(arena, in, r), fusion::instantiate(arena, in, r));
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
    LayerMath a, b;
    const int zero = a.constant(0.f);
    a.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, zero);
    fusion::gelu(b);
    const int rootA = fusion::instantiate(arena, in, a);
    const int rootB = fusion::instantiate(arena, in, b);
    ASSERT_GE(rootA, 0);
    ASSERT_GE(rootB, 0);

    std::vector<char> scratch;
    EXPECT_EQ(3, fusion::detail::markLive(arena.graph(), rootA, scratch));
    EXPECT_EQ(9, fusion::detail::markLive(arena.graph(), rootB, scratch));
    EXPECT_GT(arena.size(), (size_t)9);
}

TEST(Fusion, ExtractionYieldsAStandaloneGraph)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath a, b;
    const int zero = a.constant(0.f);
    a.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, zero);
    fusion::gelu(b);
    fusion::instantiate(arena, in, a);
    const int rootB = fusion::instantiate(arena, in, b);

    Ptr<FusionGraph> g = fusion::extract(arena.graph(), rootB, std::vector<Mat>());
    ASSERT_TRUE(g);
    EXPECT_EQ(9u, g->size());
    EXPECT_EQ(FusionEltwiseOp::INPUT, g->nodes()[0].op);
    EXPECT_EQ((int)g->size() - 1, g->outputNode);
    EXPECT_NEAR(0.5f * 1.5f * (1.f + std::erf(1.5f * 0.70710678118654752440f)),
                fusion::eval(*g, 1.5f, kNoBufs, 0), 1e-5);

    EXPECT_FALSE(fusion::extract(arena.graph(), -1, std::vector<Mat>()));
    EXPECT_FALSE(fusion::extract(arena.graph(), (int)arena.size(), std::vector<Mat>()));
}

TEST(Fusion, OverLimitConeIsRefusedNotEvaluated)
{
    FusionGraphBuilder arena;
    int cur = arena.internNode(FusionEltwiseOp::INPUT, {});
    std::vector<char> scratch;
    int steps = 0;
    while (fusion::detail::markLive(arena.graph(), cur, scratch) <= FUSION_MAX_EXPR_NODES && steps < 200) {
        LayerMath r;
        fusion::gelu(r);
        const int next = fusion::instantiate(arena, cur, r);
        ASSERT_GE(next, 0);
        cur = next;
        steps++;
    }
    ASSERT_GT(fusion::detail::markLive(arena.graph(), cur, scratch), FUSION_MAX_EXPR_NODES);
    EXPECT_FALSE(fusion::extract(arena.graph(), cur, std::vector<Mat>()));
}

TEST(Fusion, MathMatchesClosedForm)
{
    const float xs[] = { -3.f, -0.5f, 0.f, 0.25f, 1.f, 4.f };
    LayerMath r;
    for (float x : xs) {
        r = LayerMath();
        r.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, r.constant(0.f));
        EXPECT_FLOAT_EQ(std::max(x, 0.f), eval1(r, x)) << "relu " << x;

        r = LayerMath();
        r.clamp(LayerMath::INPUT_VALUE, 0.f, 6.f);
        EXPECT_FLOAT_EQ(std::min(std::max(x, 0.f), 6.f), eval1(r, x)) << "clip " << x;

        r = LayerMath(); fusion::sigmoid(r);
        EXPECT_NEAR(1.f / (1.f + std::exp(-x)), eval1(r, x), 1e-5) << "sigmoid " << x;

        r = LayerMath(); fusion::gelu(r);
        EXPECT_NEAR(0.5f * x * (1.f + std::erf(x * 0.70710678118654752440f)), eval1(r, x), 1e-5)
            << "gelu " << x;

        r = LayerMath();
        r.unary(FusionEltwiseOp::TANH, LayerMath::INPUT_VALUE);
        EXPECT_NEAR(std::tanh(x), eval1(r, x), 1e-6) << "tanh " << x;

        r = LayerMath();
        const int scaled = r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.constant(2.f));
        r.unary(FusionEltwiseOp::EXP, r.binary(FusionEltwiseOp::ADD, scaled, r.constant(5.f)));
        EXPECT_NEAR(std::exp(2.f * x + 5.f), eval1(r, x), 1e-2) << "scaled exp " << x;
    }
}

TEST(Fusion, EmptyMathIsRefused)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    EXPECT_EQ(-1, fusion::instantiate(arena, in, LayerMath()));
    EXPECT_EQ(1u, arena.size());
}

TEST(Fusion, ReversedSubIsRefused)
{
    LayerParams lp;
    lp.set("operation", "sub");
    Ptr<Layer> sub = NaryEltwiseLayer::create(lp);
    ASSERT_TRUE(sub);
    sub->inputs.assign(2, Arg());

    LayerMath r;
    ConstOperand vs;
    vs.hasValue = true;
    vs.value = 3.f;
    vs.flowIsFirstInput = false;
    EXPECT_FALSE(sub->describeMath(r, vs));

    vs.flowIsFirstInput = true;
    r = LayerMath();
    ASSERT_TRUE(sub->describeMath(r, vs));
    EXPECT_FLOAT_EQ(-1.f, eval1(r, 2.f));
}

TEST(Fusion, VariadicNaryIsRefused)
{
    LayerParams lp;
    lp.set("operation", "sum");
    Ptr<Layer> sum = NaryEltwiseLayer::create(lp);
    ASSERT_TRUE(sum);

    LayerMath r;
    ConstOperand vs;
    vs.hasValue = true;
    vs.value = 3.f;

    sum->inputs.assign(3, Arg());
    EXPECT_FALSE(sum->describeMath(r, vs));

    r = LayerMath();
    sum->inputs.assign(2, Arg());
    EXPECT_TRUE(sum->describeMath(r, vs));
}

TEST(Fusion, ClipWithOneDynamicBoundIsRefused)
{
    LayerParams lp;
    Ptr<Layer> clip = ClipLayer::create(lp);
    ASSERT_TRUE(clip);

    LayerMath r;
    ConstOperand vs;
    vs.hasValue = true;
    vs.value = 2.f;
    vs.value2 = 0.f;

    clip->inputs = { Arg(1), Arg(2) };
    EXPECT_FALSE(clip->describeMath(r, vs));

    // Clip(x, "", max): the omitted min is an empty Arg, not a missing one
    r = LayerMath();
    clip->inputs = { Arg(1), Arg(0), Arg(2) };
    EXPECT_FALSE(clip->describeMath(r, vs));

    r = LayerMath();
    vs.value2 = 6.f;
    clip->inputs = { Arg(1), Arg(2), Arg(3) };
    ASSERT_TRUE(clip->describeMath(r, vs));
    EXPECT_FLOAT_EQ(2.f, eval1(r, 1.f));
    EXPECT_FLOAT_EQ(6.f, eval1(r, 9.f));
}

TEST(Fusion, ActivationMatchRecognizesAndRefuses)
{
    int activ = ACTIV_NONE;
    std::vector<float> params;
    LayerMath r;

    r = LayerMath();
    r.clamp(LayerMath::INPUT_VALUE, 0.f, 6.f);
    ASSERT_TRUE(fusion::matchActivation(*fusion::fromMath(r), activ, params));
    EXPECT_EQ(ACTIV_CLIP, activ);
    ASSERT_EQ(2u, params.size());
    EXPECT_FLOAT_EQ(0.f, params[0]);
    EXPECT_FLOAT_EQ(6.f, params[1]);

    r = LayerMath();
    r.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, r.constant(0.f));
    ASSERT_TRUE(fusion::matchActivation(*fusion::fromMath(r), activ, params));
    EXPECT_EQ(ACTIV_RELU, activ);

    r = LayerMath(); fusion::sigmoid(r);
    ASSERT_TRUE(fusion::matchActivation(*fusion::fromMath(r), activ, params));
    EXPECT_EQ(ACTIV_SIGMOID, activ);

    r = LayerMath(); fusion::gelu(r);
    ASSERT_TRUE(fusion::matchActivation(*fusion::fromMath(r), activ, params));
    EXPECT_EQ(ACTIV_GELU, activ);

    r = LayerMath();
    r.unary(FusionEltwiseOp::SQRT, LayerMath::INPUT_VALUE);
    EXPECT_FALSE(fusion::matchActivation(*fusion::fromMath(r), activ, params));

    FusionGraphBuilder arena;
    arena.internNode(FusionEltwiseOp::INPUT, {});
    r = LayerMath();
    r.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, r.constant(0.f));
    fusion::instantiate(arena, 0, r);
    EXPECT_FALSE(fusion::matchActivation(*arena.sharedGraph(), activ, params));
}

// Matching must survive the real path: a shared arena that already holds unrelated
// nodes, sliced by fusion::extract. Comparing two fusion::fromMath() graphs cannot
// catch an ordering bug, because both sides are built the same way.
TEST(Fusion, ActivationMatchSurvivesASharedArena)
{
    struct { const char* name; void (*build)(LayerMath&); int activ; } kinds[] = {
        { "sigmoid", &fusion::sigmoid, ACTIV_SIGMOID },
        { "gelu",    &fusion::gelu,    ACTIV_GELU    },
    };

    for (const auto& k : kinds) {
        FusionGraphBuilder arena;
        const int in = arena.internNode(FusionEltwiseOp::INPUT, {});

        // an unrelated earlier chain, so the constants below are already interned
        // in an order the reference pattern does not share
        const int m1 = arena.internNode(FusionEltwiseOp::CONST, {}, -1.f);
        const int half = arena.internNode(FusionEltwiseOp::CONST, {}, 0.5f);
        arena.internNode(FusionEltwiseOp::MUL, {in, m1});
        arena.internNode(FusionEltwiseOp::MUL, {in, half});

        LayerMath m;
        k.build(m);
        const int root = fusion::instantiate(arena, in, m);
        ASSERT_GE(root, 0) << k.name;

        Ptr<FusionGraph> expr = fusion::extract(arena.graph(), root, std::vector<Mat>());
        ASSERT_TRUE(expr) << k.name;

        int activ = ACTIV_NONE;
        std::vector<float> params;
        EXPECT_TRUE(fusion::matchActivation(*expr, activ, params)) << k.name;
        EXPECT_EQ(k.activ, activ) << k.name;
    }
}

TEST(Fusion, ApplyTakesKernelPathThenInterpreterPath)
{
    LayerMath r;
    r.clamp(LayerMath::INPUT_VALUE, 0.f, 6.f);
    PreparedFusion kern;
    ASSERT_TRUE(fusion::prepare(fusion::fromMath(r), kern));
    ASSERT_TRUE(kern.activationFn != nullptr);

    int n = 5;
    Mat y(1, &n, CV_32F);
    const float src[] = { -2.f, 0.f, 3.f, 6.f, 9.f };
    std::copy(src, src + n, y.ptr<float>());
    fusion::apply(kern, y);
    const float want[] = { 0.f, 0.f, 3.f, 6.f, 6.f };
    for (int i = 0; i < n; i++)
        EXPECT_FLOAT_EQ(want[i], y.ptr<float>()[i]) << "clip i=" << i;

    r = LayerMath();
    r.unary(FusionEltwiseOp::SQRT, LayerMath::INPUT_VALUE);
    PreparedFusion interp;
    ASSERT_TRUE(fusion::prepare(fusion::fromMath(r), interp));
    EXPECT_TRUE(interp.activationFn == nullptr);

    int big = (1 << 16) + 17;
    Mat z(1, &big, CV_32F);
    for (int i = 0; i < big; i++)
        z.ptr<float>()[i] = (float)(i % 100);
    fusion::apply(interp, z);
    for (int i = 0; i < big; i += 997)
        EXPECT_NEAR(std::sqrt((float)(i % 100)), z.ptr<float>()[i], 1e-5) << "sqrt i=" << i;
}

TEST(Fusion, PerChannelConstIndexesTheLastAxis)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});
    LayerMath r;
    r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.perChannelConstant(1));
    const int root = fusion::instantiate(arena, in, r);
    ASSERT_GE(root, 0);

    int one = 1, three = 3;
    Mat b0(1, &one, CV_32F);
    b0.ptr<float>()[0] = 1.f;
    Mat b1(1, &three, CV_32F);
    b1.ptr<float>()[0] = 2.f;
    b1.ptr<float>()[1] = 3.f;
    b1.ptr<float>()[2] = 4.f;

    std::vector<Mat> tooFew(1, b0);
    EXPECT_FALSE(fusion::extract(arena.graph(), root, tooFew));

    std::vector<Mat> bufs;
    bufs.push_back(b0);
    bufs.push_back(b1);
    Ptr<FusionGraph> expr = fusion::extract(arena.graph(), root, bufs);
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
    ASSERT_TRUE(fusion::prepare(expr, fa));
    EXPECT_TRUE(fa.activationFn == nullptr);

    int sz[] = { 2, 3 };
    Mat y(2, sz, CV_32F);
    for (int i = 0; i < 6; i++)
        y.ptr<float>()[i] = 1.f;
    fusion::apply(fa, y);
    const float want[] = { 2.f, 3.f, 4.f, 2.f, 3.f, 4.f };
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(want[i], y.ptr<float>()[i]) << "i=" << i;
}

TEST(Fusion, SharedRootKeepsEachChainsOwnBuffers)
{
    FusionGraphBuilder arena;
    const int in = arena.internNode(FusionEltwiseOp::INPUT, {});

    LayerMath r;
    r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, r.perChannelConstant(0));

    const int rootA = fusion::instantiate(arena, in, r);
    const int rootB = fusion::instantiate(arena, in, r);
    ASSERT_GE(rootA, 0);
    EXPECT_EQ(rootA, rootB);

    int three = 3;
    Mat ba(1, &three, CV_32F), bb(1, &three, CV_32F);
    for (int i = 0; i < 3; i++) { ba.ptr<float>()[i] = 2.f; bb.ptr<float>()[i] = 10.f; }

    Ptr<FusionGraph> ea = fusion::extract(arena.graph(), rootA, std::vector<Mat>(1, ba));
    Ptr<FusionGraph> eb = fusion::extract(arena.graph(), rootB, std::vector<Mat>(1, bb));
    ASSERT_TRUE(ea);
    ASSERT_TRUE(eb);

    ASSERT_EQ(1u, ea->constBufs.size());
    ASSERT_EQ(1u, eb->constBufs.size());
    EXPECT_FLOAT_EQ(2.f,  ea->constBufs[0].ptr<float>()[0]);
    EXPECT_FLOAT_EQ(10.f, eb->constBufs[0].ptr<float>()[0]);

    PreparedFusion fa, fb;
    ASSERT_TRUE(fusion::prepare(ea, fa));
    ASSERT_TRUE(fusion::prepare(eb, fb));

    int sz[] = { 1, 3 };
    Mat ya(2, sz, CV_32F), yb(2, sz, CV_32F);
    for (int i = 0; i < 3; i++) { ya.ptr<float>()[i] = 1.f; yb.ptr<float>()[i] = 1.f; }
    fusion::apply(fa, ya);
    fusion::apply(fb, yb);
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(2.f,  ya.ptr<float>()[i]) << "A i=" << i;
        EXPECT_FLOAT_EQ(10.f, yb.ptr<float>()[i]) << "B i=" << i;
    }
}

}} // namespace opencv_test
