// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_FUSION_GRAPH_HPP__
#define __OPENCV_DNN_SRC_FUSION_GRAPH_HPP__
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include "opencv2/core.hpp"

namespace cv { namespace dnn {

enum { FUSION_MAX_NODES = 64 };

enum class FusionOp{
    INPUT=0,
    CONST=1,
    PER_CHANNEL_CONST=2,
    ADD=3,
    SUB=4,
    MUL=5,
    MAX=6,
    MIN=7,
    ERF=8,
    TANH=9,
    EXP=10,
    SQRT=11,
    CLAMP=12,
    RECIP=13
};

inline uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u; }

struct FusionNode{
    FusionOp op;
    std::vector<int> inputs;
    float scalar=0.0f;
    float scalar2=0.0f; //CLAMP hi bound only
    int constBufferId=-1;
};

struct FusionNodeHash{
    size_t operator()(const FusionNode& n) const noexcept {
        size_t h = std::hash<int>()((int)n.op);
        for (int i : n.inputs) h = h * 1000003u ^ (size_t)i;
        h = h * 1000003u ^ (size_t)bitsOf(n.scalar);
        h = h * 1000003u ^ (size_t)bitsOf(n.scalar2);
        h = h * 1000003u ^ (size_t)(n.constBufferId);
        return h;
    }
};

struct FusionNodeEq {
    bool operator()(const FusionNode& a, const FusionNode& b) const noexcept {
        return a.op == b.op && a.inputs == b.inputs
            && bitsOf(a.scalar)  == bitsOf(b.scalar)
            && bitsOf(a.scalar2) == bitsOf(b.scalar2)
            && a.constBufferId == b.constBufferId;
    }
};

class FusionGraph{
    public:
        const std::vector<FusionNode>& nodes() const { return nodes_; }
        size_t size() const { return nodes_.size(); }
        int outputNode=-1;
        bool empty() const { return outputNode<0;}
    private:
        std::vector<FusionNode> nodes_;
        int append(FusionNode n){
            nodes_.push_back(std::move(n));
            return (int)nodes_.size() - 1;
        }
        friend class FusionGraphBuilder;
};

class CV_EXPORTS FusionGraphBuilder{
    public:
        FusionGraph g;
        int push(FusionOp op, std::vector<int> inputs,float scalar = 0.f, float scalar2 = 0.f, int constBufferId = -1);
    private:
        std::unordered_map<FusionNode, int, FusionNodeHash, FusionNodeEq> seen;
};

struct FusionOperand{
    bool hasSide=false;
    bool perChannel=false;
    float scalar=0.f;
    float scalar2=0.f;
    int bufId=-1;
};

CV_EXPORTS int appendFusionNode(FusionGraphBuilder& b, int cur, const std::string& opType, const FusionOperand& oper);

CV_EXPORTS float evalFusionGraph(const FusionGraph& g, float x,
                               const std::vector<const float*>& constBufs, int channelIdx);

}// namespace dnn
}// namespace cv

#endif
