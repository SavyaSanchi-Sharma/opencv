// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "graph_fusion_utils.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;

struct ModelFusionPointwise
{
    explicit ModelFusionPointwise(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse() { fuseGraph(netimpl->mainGraph); }

    ConstInfoFn makeConstInfo()
    {
        return [this](Arg a, EpConstInfo& info) -> bool
        {
            if (!netimpl->isConstArg(a)) return false;
            Mat t = netimpl->argTensor(a);
            if (t.total() == 1) {
                if (t.type() == CV_32F) { info.isScalar = true; info.scalar = t.ptr<float>()[0]; return true; }
                if (t.type() == CV_64F) { info.isScalar = true; info.scalar = (float)t.ptr<double>()[0]; return true; }
                return false;
            }
            if (t.type() != CV_32F) return false;

            int nonUnit = 0;
            for (int d = 0; d < t.dims; d++)
                if (t.size[d] != 1) nonUnit++;
            if (nonUnit != 1) return false;

            info.isScalar = false;
            return true;
        };
    }

    void collect(const Ptr<Graph>& graph, vector<PointwiseChain>& chains)
    {
        vector<int> usecounts;
        netimpl->useCounts(usecounts);
        collectPointwiseChains(graph, (int)netimpl->args.size(), usecounts, makeConstInfo(), chains);
    }

    static bool dumpEnabled() { return epilogueDumpEnabled(); }

    static void dumpChain(const vector<Ptr<LayerInfo> >& prog, const PointwiseChain& ch,
                          const char* verdict, size_t accepted = 0)
    {
        std::string s;
        for (size_t k = 0; k < ch.nodes.size(); k++) {
            if (k) s += (k == accepted + 1) ? " | " : " -> ";
            s += effectiveOpType(prog[ch.nodes[k]]);
        }
        CV_LOG_INFO(NULL, cv::format("[epilogue] %-28s %s", verdict, s.c_str()));
    }

    static PointwiseChain truncateChain(const PointwiseChain& ch, size_t nsteps)
    {
        if (nsteps == ch.absorbed.size())
            return ch;
        PointwiseChain t;
        t.nodes.assign(ch.nodes.begin(), ch.nodes.begin() + nsteps + 1);
        t.absorbed.assign(ch.absorbed.begin(), ch.absorbed.begin() + nsteps);
        t.stepOperands.assign(ch.stepOperands.begin(), ch.stepOperands.begin() + nsteps);
        t.constArgs = ch.constArgs;
        t.constBufs = ch.constBufs;

        const int keep = std::min((int)nsteps, ch.epSteps);
        if (keep > 0) {
            EpBuilder b;
            int cur = b.push(EpOP::INPUT, {});
            for (int k = 0; k < keep && cur >= 0; k++)
                cur = appendNode(b, cur, effectiveOpType(ch.absorbed[k]), ch.stepOperands[k]);
            if (cur >= 0) {
                b.g.outputNode = cur;
                t.ep = b.g;
                t.epSteps = keep;
            }
        }
        return t;
    }

    void fuseGraph(Ptr<Graph>& graph)
    {
        if (!graph) return;

        const vector<Ptr<LayerInfo> >& prog = graph->prog();
        for (const Ptr<LayerInfo>& layer : prog) {
            if (!layer) continue;
            vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g : *subgraphs)
                    fuseGraph(g);
            }
        }

        vector<PointwiseChain> chains;
        collect(graph, chains);
        if (chains.empty()) return;

        const size_t nops = prog.size();
        vector<bool> dropped(nops, false);
        int nfused = 0;

        for (PointwiseChain& ch : chains) {
            const Ptr<LayerInfo>& anchor = prog[ch.nodes[0]];
            EpilogueSink* sink = dynamic_cast<EpilogueSink*>(anchor.get());
            if (!sink) {
                if (dumpEnabled()) dumpChain(prog, ch, "refused(no-sink)");
                continue;
            }

            ch.constBufs.reserve(ch.constArgs.size());
            for (Arg a : ch.constArgs)
                ch.constBufs.push_back(netimpl->argTensor(a));

            size_t accepted = 0;
            for (size_t n = ch.absorbed.size(); n >= 1; n--) {
                PointwiseChain t = truncateChain(ch, n);
                if (sink->setEpilogue(t)) { accepted = n; break; }
            }
            if (accepted == 0) {
                if (dumpEnabled()) dumpChain(prog, ch, "refused(lowering)");
                continue;
            }
            if (dumpEnabled())
                dumpChain(prog, ch, accepted == ch.absorbed.size() ? "FUSED" : "FUSED(partial)", accepted);

            const Arg finalOut = prog[ch.nodes[accepted]]->outputs[0];
            anchor->outputs[0] = finalOut;

            for (size_t k = 1; k <= accepted; k++)
                dropped[ch.nodes[k]] = true;
            nfused++;
        }

        if (nfused == 0)
            return;

        vector<Ptr<LayerInfo> > newprog;
        newprog.reserve(nops);
        for (size_t i = 0; i < nops; i++) {
            if (!dropped[i] && prog[i])
                newprog.push_back(prog[i]);
        }
        graph->setProg(newprog);

        CV_LOG_DEBUG(NULL, cv::format("fusePointwise: fused %d chain(s) in graph '%s'",
                                      nfused, graph->name().c_str()));
    }

    Net::Impl* netimpl;
};

void Net::Impl::fusePointwise()
{
    if (!getEpilogueFusionEnabled())
        return;

    ModelFusionPointwise pass(this);
    pass.fuse();
}

void graphOpTypes(Net& net, std::vector<std::string>& types)
{
    types.clear();
    Net::Impl* impl = net.getImpl();
    if (!impl || !impl->mainGraph) return;
    for (const Ptr<LayerInfo>& l : impl->mainGraph->prog())
        if (l) types.push_back(effectiveOpType(l));
}

bool graphInputShape(Net& net, MatShape& shape)
{
    Net::Impl* impl = net.getImpl();
    if (!impl || !impl->mainGraph)
        return false;

    const vector<Arg>& ins = impl->mainGraph->inputs();
    if (ins.size() != 1 || ins[0].idx <= 0 || ins[0].idx >= (int)impl->args.size())
        return false;

    const MatShape& s = impl->args[ins[0].idx].shape;
    if (s.empty() || s.size() == 0)
        return false;
    for (size_t i = 0; i < s.size(); i++)
        if (s.data()[i] <= 0)
            return false;

    shape = s;
    return true;
}

void collectPointwiseChainTypes(Net& net, std::vector<std::vector<std::string> >& chains)
{
    chains.clear();
    Net::Impl* impl = net.getImpl();
    CV_Assert(impl && impl->mainGraph);

    ModelFusionPointwise pass(impl);
    vector<PointwiseChain> found;
    pass.collect(impl->mainGraph, found);

    const vector<Ptr<LayerInfo> >& prog = impl->mainGraph->prog();
    for (const PointwiseChain& ch : found) {
        std::vector<std::string> types;
        for (int n : ch.nodes)
            types.push_back(effectiveOpType(prog[n]));
        chains.push_back(types);
    }
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
