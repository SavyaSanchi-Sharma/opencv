// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_FUSION_GRAPH_HPP__
#define __OPENCV_DNN_SRC_FUSION_GRAPH_HPP__

#include <cmath>
#include <unordered_map>
#include <vector>
#include "opencv2/dnn/dnn.hpp"

namespace cv{ namespace dnn{
CV__DNN_INLINE_NS_BEGIN

enum {
    FUSION_MAX_NODES       = 64,
    FUSION_MAX_STEP_NODES  = 16,
    FUSION_MAX_ARENA_NODES = 1 << 16
};

enum class FusionEltwiseOp
{
    INPUT = 0,
    CONST = 1,
    PER_CHANNEL_CONST = 2,
    ADD = 3,
    SUB = 4,
    MUL = 5,
    MAX = 6,
    MIN = 7,
    ERF = 8,
    TANH = 9,
    EXP = 10,
    SQRT = 11,
    CLAMP = 12,
    RECIP = 13
};

struct ValueSource
{
    bool  hasFoldedOperand = false;
    bool  flowIsFirst      = true;
    float scalar   = 0.f;
    float scalar2  = 0.f;
    int   bufferId = -1;
};

inline int arityOf(FusionEltwiseOp op)
{
    const int i = (int)op;
    CV_Assert(i >= 0 && i <= (int)FusionEltwiseOp::RECIP);
    if (i <= (int)FusionEltwiseOp::PER_CHANNEL_CONST) return 0;
    return i <= (int)FusionEltwiseOp::MIN ? 2 : 1;
}

inline unsigned bitsOf(float f) { Cv32suf s; s.f = f; return s.u; }

struct FusionRecipeNode
{
    FusionEltwiseOp op = FusionEltwiseOp::CONST;
    int   a = -1, b = -1;
    float s0 = 0.f, s1 = 0.f;
    int   buf = -1;
};

struct FusionRecipe
{
    int n = 0;
    FusionRecipeNode node[FUSION_MAX_STEP_NODES];

    bool wellFormed() const
    {
        if (n <= 0 || n > FUSION_MAX_STEP_NODES)
            return false;
        for (int i = 0; i < n; i++) {
            const FusionRecipeNode& nd = node[i];
            if (nd.op == FusionEltwiseOp::INPUT)
                return false;
            if (nd.op == FusionEltwiseOp::PER_CHANNEL_CONST && nd.buf < 0)
                return false;
            const int k = arityOf(nd.op);
            if (k >= 1 && (nd.a < -1 || nd.a >= i))
                return false;
            if (k >= 2 && (nd.b < -1 || nd.b >= i))
                return false;
        }
        return true;
    }
};

inline void unaryRecipe(FusionRecipe& r, FusionEltwiseOp op)
{
    r.node[0].op = op; r.node[0].a = -1;
    r.n = 1;
}

inline void clampRecipe(FusionRecipe& r, float lo, float hi)
{
    r.node[0].op = FusionEltwiseOp::CLAMP;
    r.node[0].a = -1; r.node[0].s0 = lo; r.node[0].s1 = hi;
    r.n = 1;
}

inline void reluRecipe(FusionRecipe& r)
{
    r.node[0].op = FusionEltwiseOp::CONST; r.node[0].s0 = 0.f;
    r.node[1].op = FusionEltwiseOp::MAX;   r.node[1].a = -1; r.node[1].b = 0;
    r.n = 2;
}

inline void sigmoidRecipe(FusionRecipe& r)
{
    r.node[0].op = FusionEltwiseOp::CONST; r.node[0].s0 =  1.f;
    r.node[1].op = FusionEltwiseOp::CONST; r.node[1].s0 = -1.f;
    r.node[2].op = FusionEltwiseOp::MUL;   r.node[2].a = -1; r.node[2].b = 1;
    r.node[3].op = FusionEltwiseOp::EXP;   r.node[3].a = 2;
    r.node[4].op = FusionEltwiseOp::ADD;   r.node[4].a = 0;  r.node[4].b = 3;
    r.node[5].op = FusionEltwiseOp::RECIP; r.node[5].a = 4;
    r.n = 6;
}

inline void geluRecipe(FusionRecipe& r)
{
    r.node[0].op = FusionEltwiseOp::CONST; r.node[0].s0 = 0.5f;
    r.node[1].op = FusionEltwiseOp::CONST; r.node[1].s0 = 1.f;
    r.node[2].op = FusionEltwiseOp::CONST; r.node[2].s0 = (float)M_SQRT1_2;
    r.node[3].op = FusionEltwiseOp::MUL;   r.node[3].a = -1; r.node[3].b = 2;
    r.node[4].op = FusionEltwiseOp::ERF;   r.node[4].a = 3;
    r.node[5].op = FusionEltwiseOp::ADD;   r.node[5].a = 1;  r.node[5].b = 4;
    r.node[6].op = FusionEltwiseOp::MUL;   r.node[6].a = 0;  r.node[6].b = -1;
    r.node[7].op = FusionEltwiseOp::MUL;   r.node[7].a = 6;  r.node[7].b = 5;
    r.n = 8;
}

inline void expRecipe(FusionRecipe& r, float normScale, float normShift)
{
    int k = 0, cur = -1;
    if (normScale != 1.f) {
        r.node[k].op = FusionEltwiseOp::CONST; r.node[k].s0 = normScale;
        const int c = k++;
        r.node[k].op = FusionEltwiseOp::MUL; r.node[k].a = cur; r.node[k].b = c;
        cur = k++;
    }
    if (normShift != 0.f) {
        r.node[k].op = FusionEltwiseOp::CONST; r.node[k].s0 = normShift;
        const int c = k++;
        r.node[k].op = FusionEltwiseOp::ADD; r.node[k].a = cur; r.node[k].b = c;
        cur = k++;
    }
    r.node[k].op = FusionEltwiseOp::EXP; r.node[k].a = cur;
    r.n = k + 1;
}

inline bool binaryConstRecipe(FusionRecipe& r, FusionEltwiseOp op, const ValueSource& side)
{
    if (!side.hasFoldedOperand)
        return false;
    if (op == FusionEltwiseOp::SUB && !side.flowIsFirst)
        return false;

    if (side.bufferId >= 0) {
        r.node[0].op = FusionEltwiseOp::PER_CHANNEL_CONST;
        r.node[0].buf = side.bufferId;
    } else {
        r.node[0].op = FusionEltwiseOp::CONST;
        r.node[0].s0 = side.scalar;
    }
    r.node[1].op = op; r.node[1].a = -1; r.node[1].b = 0;
    r.n = 2;
    return true;
}

struct FusionNode
{
    FusionEltwiseOp op = FusionEltwiseOp::INPUT;
    std::vector<int> inputs;
    float scalar = 0.f;
    float scalar2 = 0.f;
    int constBufferId = -1;

    bool operator==(const FusionNode& o) const noexcept
    {
        return op == o.op && inputs == o.inputs
            && bitsOf(scalar)  == bitsOf(o.scalar)
            && bitsOf(scalar2) == bitsOf(o.scalar2)
            && constBufferId   == o.constBufferId;
    }
};

struct FusionNodeHash
{
    size_t operator()(const FusionNode& n) const noexcept
    {
        size_t h = std::hash<int>()((int)n.op);
        for (int i : n.inputs) h = h * 1000003u ^ (size_t)i;
        h = h * 1000003u ^ (size_t)bitsOf(n.scalar);
        h = h * 1000003u ^ (size_t)bitsOf(n.scalar2);
        h = h * 1000003u ^ (size_t)n.constBufferId;
        return h;
    }
};

class FusionGraph
{
public:
    const std::vector<FusionNode>& nodes() const { return nodes_; }
    size_t size() const { return nodes_.size(); }
    int outputNode = -1;
    std::vector<Mat> constBufs;

private:
    std::vector<FusionNode> nodes_;
    int append(FusionNode n)
    {
        nodes_.push_back(std::move(n));
        return (int)nodes_.size() - 1;
    }
    friend class FusionGraphBuilder;
};

class FusionGraphBuilder
{
public:
    FusionGraphBuilder() : gp_(makePtr<FusionGraph>()) {}

    int push(FusionEltwiseOp op, std::vector<int> inputs, float scalar = 0.f,
             float scalar2 = 0.f, int constBufferId = -1)
    {
        CV_Assert((gp_->size() == 0) == (op == FusionEltwiseOp::INPUT));
        CV_Assert((int)inputs.size() == arityOf(op));
        CV_DbgAssert(gp_->size() < (size_t)FUSION_MAX_ARENA_NODES);
        for (int i : inputs)
            CV_Assert(i >= 0 && i < (int)gp_->size());

        if ((op == FusionEltwiseOp::ADD || op == FusionEltwiseOp::MUL) && inputs[0] > inputs[1])
            std::swap(inputs[0], inputs[1]);

        if (op != FusionEltwiseOp::CONST && op != FusionEltwiseOp::CLAMP) scalar = 0.f;
        if (op != FusionEltwiseOp::CLAMP)                                 scalar2 = 0.f;
        if (op != FusionEltwiseOp::PER_CHANNEL_CONST)                     constBufferId = -1;

        FusionNode cand;
        cand.op = op;
        cand.inputs = std::move(inputs);
        cand.scalar = scalar;
        cand.scalar2 = scalar2;
        cand.constBufferId = constBufferId;

        std::unordered_map<FusionNode, int, FusionNodeHash>::const_iterator it = interned_.find(cand);
        if (it != interned_.end())
            return it->second;

        const int idx = gp_->append(cand);
        interned_.emplace(std::move(cand), idx);
        return idx;
    }

    size_t size() const { return gp_->size(); }
    const FusionGraph& graph() const { return *gp_; }

    Ptr<FusionGraph> finish(int output)
    {
        CV_Assert(output == (int)gp_->size() - 1);
        gp_->outputNode = output;
        return gp_;
    }

    Ptr<FusionGraph> release() { return gp_; }

private:
    Ptr<FusionGraph> gp_;
    std::unordered_map<FusionNode, int, FusionNodeHash> interned_;
};

inline int appendFusionOp(FusionGraphBuilder& builder, int cur, const FusionRecipe& recipe)
{
    if (cur < 0 || !recipe.wellFormed())
        return -1;

    int local[FUSION_MAX_STEP_NODES];
    for (int i = 0; i < recipe.n; i++) {
        const FusionRecipeNode& nd = recipe.node[i];
        const int k = arityOf(nd.op);
        std::vector<int> inputs;
        inputs.reserve((size_t)k);
        if (k >= 1) inputs.push_back(nd.a < 0 ? cur : local[nd.a]);
        if (k >= 2) inputs.push_back(nd.b < 0 ? cur : local[nd.b]);
        local[i] = builder.push(nd.op, inputs, nd.s0, nd.s1, nd.buf);
    }
    return local[recipe.n - 1];
}

inline Ptr<FusionGraph> graphFromRecipe(const FusionRecipe& recipe)
{
    FusionGraphBuilder b;
    const int in = b.push(FusionEltwiseOp::INPUT, {});
    const int root = appendFusionOp(b, in, recipe);
    if (root != (int)b.size() - 1)
        return Ptr<FusionGraph>();
    return b.finish(root);
}

inline bool sameFusionGraph(const FusionGraph& a, const FusionGraph& b)
{
    if (a.outputNode != b.outputNode || a.size() != b.size())
        return false;
    const std::vector<FusionNode>& na = a.nodes();
    const std::vector<FusionNode>& nb = b.nodes();
    for (size_t i = 0; i < na.size(); i++) {
        if (!(na[i] == nb[i]))
            return false;
    }
    return true;
}

inline void fusionCone(const FusionGraph& g, int root, std::vector<char>& live)
{
    const std::vector<FusionNode>& nodes = g.nodes();
    CV_Assert(root >= 0 && root < (int)nodes.size());
    live.assign((size_t)root + 1, 0);
    live[root] = 1;
    for (int i = root; i >= 0; i--) {
        if (!live[i]) continue;
        for (int in : nodes[i].inputs) {
            CV_DbgAssert(in >= 0 && in < i);
            live[in] = 1;
        }
    }
}

inline int coneSize(const FusionGraph& g, int root, std::vector<char>& scratch)
{
    fusionCone(g, root, scratch);
    int n = 0;
    for (char c : scratch) n += c ? 1 : 0;
    return n;
}

inline Ptr<FusionGraph> extractSubgraph(const FusionGraph& arena, int root,
                                        const std::vector<Mat>& constBufs)
{
    const std::vector<FusionNode>& src = arena.nodes();
    if (root < 0 || root >= (int)src.size())
        return Ptr<FusionGraph>();

    std::vector<char> live;
    fusionCone(arena, root, live);
    if (!live[0] || src[0].op != FusionEltwiseOp::INPUT)
        return Ptr<FusionGraph>();

    std::vector<int> remap(live.size(), -1);
    FusionGraphBuilder out;
    for (int i = 0; i < (int)live.size(); i++) {
        if (!live[i]) continue;
        if (out.size() >= (size_t)FUSION_MAX_NODES)
            return Ptr<FusionGraph>();
        const FusionNode& n = src[i];
        if (n.op == FusionEltwiseOp::PER_CHANNEL_CONST &&
            (n.constBufferId < 0 || n.constBufferId >= (int)constBufs.size()))
            return Ptr<FusionGraph>();
        std::vector<int> ins(n.inputs.size());
        for (size_t k = 0; k < ins.size(); k++) {
            if (remap[n.inputs[k]] < 0)
                return Ptr<FusionGraph>();
            ins[k] = remap[n.inputs[k]];
        }
        remap[i] = out.push(n.op, ins, n.scalar, n.scalar2, n.constBufferId);
    }
    if (remap[root] != (int)out.size() - 1)
        return Ptr<FusionGraph>();

    Ptr<FusionGraph> g = out.finish(remap[root]);
    g->constBufs = constBufs;
    return g;
}

inline float evalFusionGraph(const FusionGraph& g, float x,
                             const std::vector<const float*>& constBufs, int channelIdx)
{
    const std::vector<FusionNode>& nodes = g.nodes();
    const int out = g.outputNode;
    CV_DbgAssert(out == (int)nodes.size() - 1);
    CV_DbgAssert(nodes.size() <= (size_t)FUSION_MAX_NODES);

    float v[FUSION_MAX_NODES];

    for (int i = 0; i <= out; i++) {
        const FusionNode& n = nodes[i];
        const float a = n.inputs.size() > 0 ? v[n.inputs[0]] : 0.f;
        const float b = n.inputs.size() > 1 ? v[n.inputs[1]] : 0.f;

        switch (n.op) {
        case FusionEltwiseOp::INPUT: v[i] = x; break;
        case FusionEltwiseOp::CONST: v[i] = n.scalar; break;
        case FusionEltwiseOp::PER_CHANNEL_CONST:
            CV_DbgAssert(n.constBufferId >= 0 && n.constBufferId < (int)constBufs.size());
            v[i] = constBufs[n.constBufferId][channelIdx];
            break;
        case FusionEltwiseOp::ADD:   v[i] = a + b; break;
        case FusionEltwiseOp::SUB:   v[i] = a - b; break;
        case FusionEltwiseOp::MUL:   v[i] = a * b; break;
        case FusionEltwiseOp::MAX:   v[i] = a < b ? b : a; break;
        case FusionEltwiseOp::MIN:   v[i] = b < a ? b : a; break;
        case FusionEltwiseOp::ERF:   v[i] = std::erf(a); break;
        case FusionEltwiseOp::TANH:  v[i] = std::tanh(a); break;
        case FusionEltwiseOp::EXP:   v[i] = std::exp(a); break;
        case FusionEltwiseOp::SQRT:  v[i] = std::sqrt(a); break;
        case FusionEltwiseOp::RECIP: v[i] = 1.f / a; break;
        case FusionEltwiseOp::CLAMP:
            v[i] = a < n.scalar ? n.scalar : (a > n.scalar2 ? n.scalar2 : a);
            break;
        }
    }
    return v[out];
}

CV__DNN_INLINE_NS_END

}// namespace dnn
}//namespace cv

#endif
