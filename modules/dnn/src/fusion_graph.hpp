// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_FUSION_GRAPH_HPP__
#define __OPENCV_DNN_SRC_FUSION_GRAPH_HPP__

#include <vector>
#include "opencv2/dnn/fusion.hpp"

namespace cv{ namespace dnn{
CV__DNN_INLINE_NS_BEGIN

inline void markReachableNodes(const FusionGraph& g, int root, std::vector<char>& live)
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

inline int reachableNodeCount(const FusionGraph& g, int root, std::vector<char>& scratch)
{
    markReachableNodes(g, root, scratch);
    int n = 0;
    for (char c : scratch) n += c ? 1 : 0;
    return n;
}

inline Ptr<FusionGraph> extractExpression(const FusionGraph& arena, int root,
                                          const std::vector<Mat>& constBufs)
{
    const std::vector<FusionNode>& src = arena.nodes();
    if (root < 0 || root >= (int)src.size())
        return Ptr<FusionGraph>();

    std::vector<char> live;
    markReachableNodes(arena, root, live);
    CV_DbgAssert(src[0].op == FusionEltwiseOp::INPUT);
    if (!live[0])
        return Ptr<FusionGraph>();

    std::vector<int> remap(live.size(), -1);
    FusionGraphBuilder out;
    for (int i = 0; i < (int)live.size(); i++) {
        if (!live[i]) continue;
        if (out.size() >= (size_t)FUSION_MAX_EXPR_NODES)
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
        remap[i] = out.internNode(n.op, ins, n.scalar, n.scalar2, n.constBufferId);
    }
    if (remap[root] != (int)out.size() - 1)
        return Ptr<FusionGraph>();

    Ptr<FusionGraph> g = out.finish(remap[root]);
    g->constBufs = constBufs;
    return g;
}

CV__DNN_INLINE_NS_END
}// namespace dnn
}//namespace cv

#endif
