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
#include "epilogue.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

enum class OpShape { ANCHOR_TEMPLATE, ANCHOR_REDUCE, MAP, UNCLASSIFIED };

CV_EXPORTS OpShape classify(const std::string& layerType);

CV_EXPORTS std::string effectiveOpType(const Ptr<LayerInfo>& l);

CV_EXPORTS void buildProducerOf(const Ptr<Graph>& g, int nargs, std::vector<int>& producerOf);

struct EpConstInfo
{
    bool isScalar = false;
    float scalar = 0.f;
};

typedef std::function<bool(Arg, EpConstInfo&)> ConstInfoFn;

struct PointwiseChain
{
    std::vector<int> nodes;
    std::vector<Ptr<LayerInfo> > absorbed;
    std::vector<Arg> constArgs;
    std::vector<Mat> constBufs;
    std::vector<EpOperand> stepOperands;
    EpGraph ep;
    int epSteps = 0;
};

struct CV_EXPORTS EpilogueSink
{
    virtual ~EpilogueSink();
    virtual bool setEpilogue(const PointwiseChain& ch) = 0;
};

CV_EXPORTS bool getEpilogueFusionEnabled();
CV_EXPORTS void setEpilogueFusionEnabled(bool enabled);

CV_EXPORTS bool epilogueDumpEnabled();

CV_EXPORTS bool epilogueInterpEnabled();
CV_EXPORTS void setEpilogueInterpEnabled(bool enabled);

CV_EXPORTS void collectPointwiseChains(const Ptr<Graph>& g, int nargs,
                                        const std::vector<int>& useCounts,
                                        const ConstInfoFn& constInfo,
                                        std::vector<PointwiseChain>& chains);

CV_EXPORTS void collectPointwiseChainTypes(Net& net, std::vector<std::vector<std::string> >& chains);

CV_EXPORTS void graphOpTypes(Net& net, std::vector<std::string>& types);

CV_EXPORTS bool graphInputShape(Net& net, MatShape& shape);

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
