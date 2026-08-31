// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_FUSION_HPP
#define OPENCV_DNN_FUSION_HPP

#include <cmath>
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>
#include "../dnn/version.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
//! @addtogroup dnn
//! @{

enum {
    FUSION_MAX_EXPR_NODES   = 64,      //!< cap on one extracted expression
    FUSION_MAX_RECIPE_NODES = 16,      //!< cap on one layer's recipe
    FUSION_MAX_ARENA_NODES  = 1 << 16  //!< cap on the shared arena for a whole graph
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

inline int operandCount(FusionEltwiseOp op)
{
    const int i = (int)op;
    CV_Assert(i >= 0 && i <= (int)FusionEltwiseOp::RECIP);
    if (i <= (int)FusionEltwiseOp::PER_CHANNEL_CONST) return 0;
    return i <= (int)FusionEltwiseOp::MIN ? 2 : 1;
}

inline unsigned floatBits(float f) { Cv32suf s; s.f = f; return s.u; }

/** @brief The constant operand(s) a layer has alongside the value flowing into it,
 *  e.g. the 3 in `x + 3`, or Clip's two bounds. A layer with none gets hasValue false.
 */
struct ConstOperand
{
    bool  hasValue         = false;  //!< false means there is no constant operand
    bool  flowIsFirstInput = true;   //!< the flowing value is the layer's input 0
    float value            = 0.f;    //!< scalar constant, or Clip's lower bound
    float value2           = 0.f;    //!< Clip's upper bound; unused otherwise
    int   bufferId         = -1;     //!< >=0 for a per-channel constant, indexes constBufs
};

struct FusionRecipeNode
{
    FusionEltwiseOp op = FusionEltwiseOp::CONST;
    int   left = -1, right = -1;
    float scalar = 0.f, scalar2 = 0.f;
    int   bufferId = -1;
};

/** @brief Straight-line description of one layer's elementwise math.
 *
 * A layer builds its recipe by appending nodes; each call returns the index of
 * the node it added, and an operand is either one of those indices or
 * INPUT_VALUE, standing for the value arriving from the layer being fused into.
 * Since nodes can only be added this way, a recipe is always well formed.
 */
struct FusionRecipe
{
    enum { INPUT_VALUE = -1 };

    int nodeCount() const { return nodeCount_; }
    const FusionRecipeNode& nodeAt(int i) const
    {
        CV_DbgAssert(i >= 0 && i < nodeCount_);
        return nodes_[i];
    }

    int constant(float value)
    { return appendNode(FusionEltwiseOp::CONST, INPUT_VALUE, INPUT_VALUE, value, 0.f, -1); }

    int perChannelConstant(int bufferId)
    {
        CV_Assert(bufferId >= 0);
        return appendNode(FusionEltwiseOp::PER_CHANNEL_CONST, INPUT_VALUE, INPUT_VALUE,
                          0.f, 0.f, bufferId);
    }

    int unary(FusionEltwiseOp op, int operand)
    {
        CV_Assert(operandCount(op) == 1);
        return appendNode(op, operand, INPUT_VALUE, 0.f, 0.f, -1);
    }

    int binary(FusionEltwiseOp op, int left, int right)
    {
        CV_Assert(operandCount(op) == 2);
        return appendNode(op, left, right, 0.f, 0.f, -1);
    }

    int clamp(int operand, float lo, float hi)
    { return appendNode(FusionEltwiseOp::CLAMP, operand, INPUT_VALUE, lo, hi, -1); }

private:
    int appendNode(FusionEltwiseOp op, int a, int b, float s0, float s1, int buf)
    {
        CV_Assert(nodeCount_ < FUSION_MAX_RECIPE_NODES);
        CV_Assert(a >= INPUT_VALUE && a < nodeCount_);
        CV_Assert(b >= INPUT_VALUE && b < nodeCount_);
        FusionRecipeNode& nd = nodes_[nodeCount_];
        nd.op = op; nd.left = a; nd.right = b; nd.scalar = s0; nd.scalar2 = s1; nd.bufferId = buf;
        return nodeCount_++;
    }

    int nodeCount_ = 0;
    FusionRecipeNode nodes_[FUSION_MAX_RECIPE_NODES];
};

// Sigmoid and Gelu are each built twice - by their own layer, and by the
// reference table matchKnownActivation compares against - and the two have to
// emit nodes in the same order, so they share one definition.
inline void sigmoidRecipe(FusionRecipe& r)
{
    const int one      = r.constant(1.f);
    const int minusOne = r.constant(-1.f);
    const int negated  = r.binary(FusionEltwiseOp::MUL, FusionRecipe::INPUT_VALUE, minusOne);
    const int exponent = r.unary(FusionEltwiseOp::EXP, negated);
    const int denom    = r.binary(FusionEltwiseOp::ADD, one, exponent);
    r.unary(FusionEltwiseOp::RECIP, denom);
}

inline void geluRecipe(FusionRecipe& r)
{
    const int half     = r.constant(0.5f);
    const int one      = r.constant(1.f);
    const int invSqrt2 = r.constant((float)M_SQRT1_2);
    const int scaled   = r.binary(FusionEltwiseOp::MUL, FusionRecipe::INPUT_VALUE, invSqrt2);
    const int erfTerm  = r.unary(FusionEltwiseOp::ERF, scaled);
    const int gate     = r.binary(FusionEltwiseOp::ADD, one, erfTerm);
    const int halfX    = r.binary(FusionEltwiseOp::MUL, half, FusionRecipe::INPUT_VALUE);
    r.binary(FusionEltwiseOp::MUL, halfX, gate);
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
            && floatBits(scalar)  == floatBits(o.scalar)
            && floatBits(scalar2) == floatBits(o.scalar2)
            && constBufferId   == o.constBufferId;
    }
};

struct FusionNodeHash
{
    size_t operator()(const FusionNode& n) const noexcept
    {
        size_t h = std::hash<int>()((int)n.op);
        for (int i : n.inputs) h = h * 1000003u ^ (size_t)i;
        h = h * 1000003u ^ (size_t)floatBits(n.scalar);
        h = h * 1000003u ^ (size_t)floatBits(n.scalar2);
        h = h * 1000003u ^ (size_t)n.constBufferId;
        return h;
    }
};

class CV_EXPORTS FusionGraph
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

class CV_EXPORTS FusionGraphBuilder
{
public:
    FusionGraphBuilder() : gp_(makePtr<FusionGraph>()) {}

    /** @brief Adds one node, reusing an identical existing one if there is
     *  already one in this graph. Returns the node's index either way. */
    int internNode(FusionEltwiseOp op, std::vector<int> inputs, float scalar = 0.f,
                   float scalar2 = 0.f, int constBufferId = -1)
    {
        CV_Assert((gp_->size() == 0) == (op == FusionEltwiseOp::INPUT));
        CV_Assert((int)inputs.size() == operandCount(op));
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

    //! The graph as built so far; the builder keeps writing to it.
    Ptr<FusionGraph> sharedGraph() { return gp_; }

private:
    Ptr<FusionGraph> gp_;
    std::unordered_map<FusionNode, int, FusionNodeHash> interned_;
};

inline int instantiateRecipe(FusionGraphBuilder& builder, int inputNode,
                             const FusionRecipe& recipe)
{
    if (inputNode < 0 || recipe.nodeCount() <= 0)
        return -1;

    int graphIndex[FUSION_MAX_RECIPE_NODES];
    for (int i = 0; i < recipe.nodeCount(); i++) {
        const FusionRecipeNode& nd = recipe.nodeAt(i);
        const int k = operandCount(nd.op);
        std::vector<int> inputs;
        inputs.reserve((size_t)k);
        if (k >= 1) inputs.push_back(nd.left  < 0 ? inputNode : graphIndex[nd.left]);
        if (k >= 2) inputs.push_back(nd.right < 0 ? inputNode : graphIndex[nd.right]);
        graphIndex[i] = builder.internNode(nd.op, inputs, nd.scalar, nd.scalar2, nd.bufferId);
    }
    return graphIndex[recipe.nodeCount() - 1];
}

inline Ptr<FusionGraph> patternFromRecipe(const FusionRecipe& recipe)
{
    FusionGraphBuilder b;
    const int in = b.internNode(FusionEltwiseOp::INPUT, {});
    const int root = instantiateRecipe(b, in, recipe);
    if (root != (int)b.size() - 1)
        return Ptr<FusionGraph>();
    return b.finish(root);
}

inline bool structurallyEqual(const FusionGraph& a, const FusionGraph& b)
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

inline float evalFusionGraph(const FusionGraph& g, float x,
                             const std::vector<const float*>& constBufs, int channelIdx)
{
    const std::vector<FusionNode>& nodes = g.nodes();
    const int out = g.outputNode;
    CV_DbgAssert(out == (int)nodes.size() - 1);
    CV_DbgAssert(nodes.size() <= (size_t)FUSION_MAX_EXPR_NODES);

    float v[FUSION_MAX_EXPR_NODES];

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

//! @}
CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif  // OPENCV_DNN_FUSION_HPP
