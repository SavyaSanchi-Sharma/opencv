// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_GRAPH_FUSION_UTILS_HPP__
#define __OPENCV_DNN_SRC_GRAPH_FUSION_UTILS_HPP__
#include <functional>
#include <string>
#include <vector>
#include "opencv2/dnn/dnn.hpp"
#include "fusion_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

enum class OpShape { ANCHOR_TEMPLATE, ANCHOR_REDUCE, MAP, UNCLASSIFIED };

CV_EXPORTS OpShape classify(const std::string& layerType);

CV_EXPORTS std::string effectiveOpType(const Ptr<LayerInfo>& l);

CV_EXPORTS void buildProducerOf(const Ptr<Graph>& g, int nargs, std::vector<int>& producerOf);

struct FusionConstInfo
{
    bool isScalar = false;
    float scalar = 0.f;
};

typedef std::function<bool(Arg, FusionConstInfo&)> ConstInfoFn;

struct AgnosticChain
{
    std::vector<int> nodes;
    std::vector<Ptr<LayerInfo> > absorbed;
    std::vector<Arg> constArgs;
    std::vector<Mat> constBufs;
    std::vector<FusionOperand> stepOperands;
    FusionGraph fg;
    int fgSteps = 0;
};

struct CV_EXPORTS FusionSink
{
    virtual ~FusionSink();
    virtual bool setFusion(const AgnosticChain& ch) = 0;
};

CV_EXPORTS bool getAgnosticFusionEnabled();
CV_EXPORTS void setAgnosticFusionEnabled(bool enabled);

CV_EXPORTS bool fusionDumpEnabled();

CV_EXPORTS bool fusionInterpEnabled();
CV_EXPORTS void setFusionInterpEnabled(bool enabled);

CV_EXPORTS void collectAgnosticChains(const Ptr<Graph>& g, int nargs,
                                        const std::vector<int>& useCounts,
                                        const ConstInfoFn& constInfo,
                                        std::vector<AgnosticChain>& chains);

CV_EXPORTS void collectAgnosticChainTypes(Net& net, std::vector<std::vector<std::string> >& chains);

CV_EXPORTS void graphOpTypes(Net& net, std::vector<std::string>& types);

CV_EXPORTS bool graphInputShape(Net& net, MatShape& shape);

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
