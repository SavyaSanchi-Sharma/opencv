// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Folds a Transpose-of-last-two-dims into the consuming MatMul's trans_a / trans_b.
//
// Pattern:
//   X -> Transpose(perm=[0,..,n-3, n-1, n-2]) -> MatMul(., Y)
//   =>  X -> MatMul(trans_a=true, ., Y)
//
// Symmetric on the second operand -> trans_b.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

namespace {

bool isLastTwoSwap(const vector<int>& perm)
{
    int n = (int)perm.size();
    if (n < 2) return false;
    for (int i = 0; i < n - 2; i++)
        if (perm[i] != i) return false;
    return perm[n - 2] == n - 1 && perm[n - 1] == n - 2;
}

bool tryFoldTransposeIntoMatMul(LayerInfo* layerInfo, int slot, LayerInfo* producerInfo)
{
    if (slot > 1) return false;
    MatMulLayer* mm = dynamic_cast<MatMulLayer*>(layerInfo);
    if (!mm || layerInfo->inputs.size() != 2) return false;

    TransposeLayer* tr = dynamic_cast<TransposeLayer*>(producerInfo);
    if (!tr || producerInfo->outputs.size() != 1) return false;
    if (!isLastTwoSwap(tr->perm)) return false;

    if (slot == 0) mm->trans_a = !mm->trans_a;
    else           mm->trans_b = !mm->trans_b;

    layerInfo->inputs[slot] = producerInfo->inputs[0];
    return true;
}

} // namespace

void Net::Impl::fuseTransposeMatMul()
{
    if (!mainGraph)
        return;
    vector<int> usecounts;
    useCounts(usecounts);
    foldSingleUseProducers(*this, mainGraph, usecounts, tryFoldTransposeIntoMatMul);
}

CV__DNN_INLINE_NS_END
}}
