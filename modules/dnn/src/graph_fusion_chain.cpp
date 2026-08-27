// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "fusion_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;

namespace {

static void firstConsumerOf(const vector<Ptr<LayerInfo> >& prog, int nargs,
                            vector<int>& firstConsumer)
{
    firstConsumer.assign((size_t)nargs, -1);
    for (size_t j = 0; j < prog.size(); j++) {
        if (!prog[j]) continue;
        for (Arg in : prog[j]->inputs) {
            if (in.idx > 0 && in.idx < nargs && firstConsumer[in.idx] < 0)
                firstConsumer[in.idx] = (int)j;
        }
    }
}

class ChainFuser
{
public:
    ChainFuser(Net::Impl& net, const Ptr<Graph>& graph, const vector<int>& usecounts)
        : net_(net), graph_(graph), usecounts_(usecounts)
    {
        CV_Assert((int)usecounts_.size() == (int)net_.args.size());
        taken_.assign(prog().size(), false);
        CV_Assert(arena_.push(FusionEltwiseOp::INPUT, {}) == 0);
    }
    ChainFuser(const ChainFuser&) = delete;
    ChainFuser& operator=(const ChainFuser&) = delete;

    bool fuse()
    {
        collectChains();
        if (chains_.empty())
            return false;
        freezeArena();
        negotiate();
        if (nfused_ == 0)
            return false;
        rewriteProg();
        return true;
    }

private:
    struct Candidate
    {
        vector<int> progIdx;
        vector<Arg> constArgs;
        vector<int> stepRoots;
        vector<Mat> constBufs;
    };

    struct Walk { int root = 0; bool open = true; };

    const vector<Ptr<LayerInfo> >& prog() const { return graph_->prog(); }

    static int internConstArg(Candidate& c, Arg a)
    {
        for (size_t k = 0; k < c.constArgs.size(); k++) {
            if (c.constArgs[k].idx == a.idx)
                return (int)k;
        }
        c.constArgs.push_back(a);
        return (int)c.constArgs.size() - 1;
    }

    bool foldableConst(Arg a, bool& isScalar, float& scalarVal) const
    {
        if (!net_.isConstArg(a))
            return false;
        Mat t = net_.argTensor(a);
        if (t.total() == 1) {
            if (t.type() == CV_32F) { isScalar = true; scalarVal = t.ptr<float>()[0]; return true; }
            if (t.type() == CV_64F) { isScalar = true; scalarVal = (float)t.ptr<double>()[0]; return true; }
            return false;
        }
        if (t.type() != CV_32F)
            return false;
        int nonUnit = 0;
        for (int d = 0; d < t.dims; d++)
            if (t.size[d] != 1) nonUnit++;
        if (nonUnit != 1)
            return false;
        isScalar = false;
        return true;
    }

    bool readValueSource(const Ptr<LayerInfo>& L, Arg cur, Candidate& c, ValueSource& vs) const
    {
        vector<Arg> side;
        for (Arg in : L->inputs) {
            if (in.idx == cur.idx || in.idx == 0)
                continue;
            side.push_back(in);
        }
        if (side.empty())
            return true;
        if (side.size() > 2)
            return false;

        vs.flowIsFirst = !L->inputs.empty() && L->inputs[0].idx == cur.idx;

        if (side.size() == 2) {
            bool s0 = false, s1 = false;
            float v0 = 0.f, v1 = 0.f;
            if (!foldableConst(side[0], s0, v0) || !s0)
                return false;
            if (!foldableConst(side[1], s1, v1) || !s1)
                return false;
            vs.hasFoldedOperand = true;
            vs.scalar = v0;
            vs.scalar2 = v1;
            return true;
        }

        bool isScalar = false;
        float scalarVal = 0.f;
        if (!foldableConst(side[0], isScalar, scalarVal))
            return false;
        vs.hasFoldedOperand = true;
        if (isScalar)
            vs.scalar = scalarVal;
        else
            vs.bufferId = internConstArg(c, side[0]);
        return true;
    }

    bool describesItself(const Ptr<LayerInfo>& L) const
    {
        Layer* l = dynamic_cast<Layer*>(L.get());
        if (!l)
            return false;
        FusionRecipe r;
        ValueSource vs;
        return l->describeMath(r, vs);
    }

    void growChain(size_t anchor, Candidate& c)
    {
        CV_Assert(!arenaPtr_);

        Walk w;
        Arg curArg = prog()[anchor]->outputs[0];
        int prod = (int)anchor;

        while (w.open && curArg.idx > 0 && curArg.idx < (int)usecounts_.size() &&
               usecounts_[curArg.idx] == 1) {
            const int j = firstConsumer_[curArg.idx];
            if (j <= prod || j >= (int)prog().size() || taken_[j])
                break;

            const Ptr<LayerInfo>& L = prog()[j];
            if (!L || L->outputs.size() != 1 || L->subgraphs())
                break;
            Layer* l = dynamic_cast<Layer*>(L.get());
            if (!l)
                break;

            if (arena_.size() >= (size_t)FUSION_MAX_ARENA_NODES)
                break;

            const size_t savedSlots = c.constArgs.size();
            ValueSource vs;
            FusionRecipe r;
            if (!readValueSource(L, curArg, c, vs) || !l->describeMath(r, vs)) {
                c.constArgs.resize(savedSlots);
                break;
            }

            const int next = appendFusionOp(arena_, w.root, r);
            if (next < 0 || coneSize(arena_.graph(), next, coneScratch_) > FUSION_MAX_NODES) {
                c.constArgs.resize(savedSlots);
                break;
            }

            CV_DbgAssert(next > w.root);
            w.root = next;
            c.stepRoots.push_back(next);
            c.progIdx.push_back(j);
            curArg = L->outputs[0];
            prod = j;
        }
    }

    void collectChains()
    {
        const int nargs = (int)net_.args.size();
        firstConsumerOf(prog(), nargs, firstConsumer_);

        for (size_t i = 0; i < prog().size(); i++) {
            const Ptr<LayerInfo>& L = prog()[i];
            if (!L || taken_[i])
                continue;
            if (L->outputs.size() != 1 || L->subgraphs())
                continue;
            if (describesItself(L))
                continue;
            if (!dynamic_cast<Layer*>(L.get()))
                continue;

            Candidate c;
            c.progIdx.push_back((int)i);
            growChain(i, c);

            if (c.stepRoots.empty())
                continue;
            for (int n : c.progIdx)
                taken_[n] = true;
            chains_.push_back(c);
        }
    }

    void freezeArena()
    {
        arenaPtr_ = arena_.release();
        for (Candidate& c : chains_) {
            c.constBufs.reserve(c.constArgs.size());
            for (Arg a : c.constArgs)
                c.constBufs.push_back(net_.argTensor(a));
        }
    }

    Ptr<FusionGraph> exprFor(int root, const vector<Mat>& bufs) const
    {
        return extractSubgraph(*arenaPtr_, root, bufs);
    }

    void negotiate()
    {
        dropped_.assign(prog().size(), false);

        for (Candidate& c : chains_) {
            const Ptr<LayerInfo>& anchorInfo = prog()[c.progIdx[0]];
            Layer* sink = dynamic_cast<Layer*>(anchorInfo.get());
            if (!sink)
                continue;

            size_t accepted = 0;
            for (size_t n = c.stepRoots.size(); n >= 1; n--) {
                Ptr<FusionGraph> expr = exprFor(c.stepRoots[n - 1], c.constBufs);
                if (!expr)
                    continue;
                if (sink->tryFuseChain(expr)) {
                    accepted = n;
                    break;
                }
            }
            if (accepted == 0) {
                CV_LOG_DEBUG(NULL, cv::format("[fusion] refused %s (+%d)",
                                              anchorInfo->type.c_str(),
                                              (int)c.stepRoots.size()));
                continue;
            }

            anchorInfo->outputs[0] = prog()[c.progIdx[accepted]]->outputs[0];
            for (size_t k = 1; k <= accepted; k++)
                dropped_[c.progIdx[k]] = true;
            nfused_++;

            CV_LOG_DEBUG(NULL, cv::format("[fusion] FUSED %s +%d of %d",
                                          anchorInfo->type.c_str(),
                                          (int)accepted, (int)c.stepRoots.size()));
        }
    }

    void rewriteProg()
    {
        const size_t nops = prog().size();
        vector<Ptr<LayerInfo> > newprog;
        newprog.reserve(nops);
        for (size_t i = 0; i < nops; i++) {
            if (!dropped_[i] && prog()[i])
                newprog.push_back(prog()[i]);
        }
        graph_->setProg(newprog);

        CV_LOG_DEBUG(NULL, cv::format("fuseLayer: fused %d chain(s) in graph '%s', arena %d nodes",
                                      nfused_, graph_->name().c_str(), (int)arenaPtr_->size()));
    }

    Net::Impl& net_;
    const Ptr<Graph>& graph_;
    const vector<int>& usecounts_;

    FusionGraphBuilder arena_;
    Ptr<FusionGraph>   arenaPtr_;
    vector<int>        firstConsumer_;
    vector<bool>       taken_, dropped_;
    vector<Candidate>  chains_;
    vector<char>       coneScratch_;
    int nfused_ = 0;
};

static bool fuseChainsInGraph(Net::Impl& net, const Ptr<Graph>& graph,
                             const vector<int>& usecounts)
{
    if (!graph)
        return false;

    bool subFused = false;
    for (const Ptr<LayerInfo>& layer : graph->prog()) {
        if (!layer) continue;
        if (vector<Ptr<Graph> >* subs = layer->subgraphs()) {
            for (Ptr<Graph>& g : *subs) {
                if (fuseChainsInGraph(net, g, usecounts))
                    subFused = true;
            }
        }
    }

    vector<int> recounted;
    if (subFused)
        net.useCounts(recounted);

    ChainFuser fuser(net, graph, subFused ? recounted : usecounts);
    const bool fusedHere = fuser.fuse();
    return fusedHere || subFused;
}

} // namespace

void Net::Impl::fuseLayer()
{
    if (!mainGraph)
        return;
    vector<int> usecounts;
    useCounts(usecounts);
    fuseChainsInGraph(*this, mainGraph, usecounts);
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
