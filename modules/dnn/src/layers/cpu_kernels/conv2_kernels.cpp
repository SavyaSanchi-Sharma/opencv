// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "../../net_impl.hpp"
#include "../conv2_common.hpp"
#include "conv2_kernels.simd.hpp"
#include "layers/cpu_kernels/conv2_kernels.simd_declarations.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Scalar conv for FP16/BF16: true half arithmetic, every step rounded to 16 bits.
template <typename _Tp>
static void conv16xfC8(const void* inp__, const void* residual__, void* out__,
                       const ConvState& cs, const void* weights__,
                       const float* scale__, const float* bias__)
{
    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    CV_Assert_N(inpshape.layout == DATA_LAYOUT_BLOCK, outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.nspatialdims <= ConvState::MAX_CONV_DIMS);
    CV_Assert(cs.wshape.dims == 5);

    const int sdims = cs.nspatialdims;
    const int C0 = inpshape.back(), K0 = outshape.back();
    CV_Assert(C0 == K0);

    const int N = outshape[0];
    const int C = inpshape.channels(), K = outshape.channels();
    const int C1 = inpshape[1], K1 = outshape[1];

    const int Di = sdims > 2 ? inpshape[sdims - 1] : 1;
    const int Hi = sdims > 1 ? inpshape[sdims] : 1;
    const int Wi = inpshape[sdims + 1];
    const int D  = sdims > 2 ? outshape[sdims - 1] : 1;
    const int H  = sdims > 1 ? outshape[sdims] : 1;
    const int W  = outshape[sdims + 1];

    const int ngroups = cs.ngroups;
    const int Cg = C / ngroups, Kg = K / ngroups;
    const int Kblk = cs.wshape[1], ksize = cs.wshape[2], C1Max = cs.wshape[3];

    const int SZ = cs.strides[0], SY = cs.strides[1], SX = cs.strides[2];
    const int padZ0 = cs.pads[0], padY0 = cs.pads[1], padX0 = cs.pads[2];
    const int* coordtab = cs.coordtab.data();

    const _Tp* inp = (const _Tp*)inp__;
    const _Tp* res = (const _Tp*)residual__;
    _Tp* out = (_Tp*)out__;
    const _Tp* wdata = (const _Tp*)weights__;

    // Same epilogue coefficients the SIMD kernels build via setupActivation().
    const FastActivation fastActivation = cs.fastActivation;
    const float* activParams = cs.activParams.data();
    float maxval = FLT_MAX, defaultAlpha = 0.f;
    if (fastActivation == FAST_ACTIV_CLIP) {
        CV_Assert(cs.activParams.size() == 2u);
        maxval = activParams[1];
    } else if (fastActivation == FAST_ACTIV_LEAKY_RELU) {
        CV_Assert(cs.activParams.size() == 1u);
        defaultAlpha = activParams[0];
    } else if (fastActivation == FAST_ACTIV_PRELU) {
        CV_Assert(cs.activParams.size() == size_t(K));
    } else if (fastActivation == FAST_ACTIV_NONE) {
        defaultAlpha = 1.f;
    }

    const size_t inpplane = (size_t)Di*Hi*Wi*C0;
    const size_t outplane = (size_t)D*H*W*K0;
    const size_t wKblkStride = (size_t)ksize*C1Max*C0*K0;
    const size_t outtotal = outshape.total();

    // Lanes past K in the last output block are never written below.
    if (K % K0 != 0)
        memset(out__, 0, outtotal*sizeof(_Tp));

    // One task per (n, group, K-block): the K0 output channels of a block are the
    // innermost, contiguous axis of the packed weights, so accumulating all of them
    // together keeps the weight reads sequential and reuses each input load K0 times.
    parallel_for_(Range(0, N*ngroups*Kblk), [&](const Range& r) {
        std::vector<float> accbuf(K0);

        for (int task = r.start; task < r.end; task++) {
            const int n = task / (ngroups*Kblk);
            const int rem = task - n*(ngroups*Kblk);
            const int g = rem / Kblk;
            const int kblk = rem - g*Kblk;

            const int c1base = (g*Cg) / C0;
            const int c00 = (g*Cg) & (C0 - 1);
            const int k0end = std::min(K0, Kg - kblk*K0);

            const _Tp* wblk = wdata + (size_t)(g*Kblk + kblk)*wKblkStride;

            for (int z0 = 0; z0 < D; z0++) {
                const int zi_ = z0*SZ - padZ0;
                for (int y0 = 0; y0 < H; y0++) {
                    const int yi_ = y0*SY - padY0;
                    for (int x0 = 0; x0 < W; x0++) {
                        const int xi_ = x0*SX - padX0;
                        for (int k0 = 0; k0 < K0; k0++)
                            accbuf[k0] = 0.f;

                        for (int i = 0; i < ksize; i++) {
                            const int zi = zi_ + coordtab[i*ConvState::MAX_CONV_DIMS];
                            const int yi = yi_ + coordtab[i*ConvState::MAX_CONV_DIMS + 1];
                            const int xi = xi_ + coordtab[i*ConvState::MAX_CONV_DIMS + 2];
                            if ((unsigned)zi >= (unsigned)Di ||
                                (unsigned)yi >= (unsigned)Hi ||
                                (unsigned)xi >= (unsigned)Wi)
                                continue;

                            const size_t spat = (size_t)((zi*Hi + yi)*Wi + xi)*C0;
                            const _Tp* wi = wblk + (size_t)i*C1Max*C0*K0;

                            for (int c = 0; c < Cg; c++) {
                                const int ch = c00 + c;
                                const int c1w = ch / C0, c0w = ch & (C0 - 1);
                                const float av =
                                    (float)inp[((size_t)n*C1 + c1base + c1w)*inpplane + spat + c0w];
                                const _Tp* w = wi + (size_t)c1w*C0*K0 + (size_t)c0w*K0;
                                // One rounding per MAC, as a fused half FMA would give.
                                for (int k0 = 0; k0 < K0; k0++)
                                    accbuf[k0] = (float)_Tp(accbuf[k0] + av*(float)w[k0]);
                            }
                        }

                        const size_t spatofs = (size_t)((z0*H + y0)*W + x0)*K0;
                        for (int k0 = 0; k0 < k0end; k0++) {
                            const int k = g*Kg + kblk*K0 + k0;
                            const size_t outofs =
                                ((size_t)n*K1 + k/K0)*outplane + (k % K0) + spatofs;

                            float acc = accbuf[k0];
                            acc = (float)_Tp(acc*(scale__ ? scale__[k] : 1.f) +
                                             (bias__ ? bias__[k] : 0.f));
                            if (res)
                                acc = (float)_Tp(acc + (float)res[outofs]);
                            const float alpha = fastActivation == FAST_ACTIV_PRELU
                                                ? activParams[k] : defaultAlpha;
                            acc = acc >= 0.f ? acc : acc*alpha;
                            out[outofs] = _Tp(std::min(acc, maxval));
                        }
                    }
                }
            }
        }
    });

    // A generic (non-fast) activation only has float kernels, so run it as a
    // post-pass over small tiles rather than widening the whole tensor.
    if (cs.activation) {
        const int TILE = 1024;
        const int ntiles = (int)((outtotal + TILE - 1)/TILE);
        parallel_for_(Range(0, ntiles), [&](const Range& r) {
            std::vector<float> buf(TILE);
            for (int t = r.start; t < r.end; t++) {
                const size_t start = (size_t)t*TILE;
                const size_t len = std::min((size_t)TILE, outtotal - start);
                for (size_t i = 0; i < len; i++)
                    buf[i] = (float)out[start + i];
                cs.activation(buf.data(), buf.data(), len, activParams);
                for (size_t i = 0; i < len; i++)
                    out[start + i] = _Tp(buf[i]);
            }
        });
    }
}

static void conv16fC8(const void* inp, const void* residual, void* out,
                      const ConvState& cs, const void* weights,
                      const float* scale, const float* bias)
{
    conv16xfC8<hfloat>(inp, residual, out, cs, weights, scale, bias);
}

static void conv16bfC8(const void* inp, const void* residual, void* out,
                       const ConvState& cs, const void* weights,
                       const float* scale, const float* bias)
{
    conv16xfC8<bfloat>(inp, residual, out, cs, weights, scale, bias);
}

ConvFunc getConvFunc(int depth, int C0)
{
    if (C0 == 8) {
        if (depth == CV_16F) return conv16fC8;
        if (depth == CV_16BF) return conv16bfC8;
    }
    CV_CPU_DISPATCH(getConvFunc_, (depth, C0), CV_CPU_DISPATCH_MODES_ALL);
}

CV__DNN_INLINE_NS_END
}}
