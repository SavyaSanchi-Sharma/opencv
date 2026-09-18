// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "../net_impl.hpp"
#include "../op_cuda.hpp"

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/trilu.hpp"
#endif
using namespace std;
namespace cv { namespace dnn {

class TriluLayerImpl CV_FINAL : public TriluLayer {
    public:
        TriluLayerImpl(const LayerParams &params) {
            setParamsFrom(params);
            upperTri = params.get<bool>("upper", true);
        }

        virtual bool supportBackend(int backendId) CV_OVERRIDE
        {
#ifdef HAVE_CUDA
            if (backendId == DNN_BACKEND_CUDA)
                return cudaSupported();
#endif
            return backendId == DNN_BACKEND_OPENCV;
        }

#ifdef HAVE_CUDA
        /* k arrives as an int64 input tensor; the CUDA path bakes it in at init, so a
         * non-const k stays on the CPU. */
        bool cudaSupported() const
        {
            Net::Impl* netimpl_ = getNetImpl(this);
            if (!netimpl_ || this->inputs.empty())
                return false;

            const int t = netimpl_->argType(this->inputs[0]);
            if (t != CV_32F && t != CV_16F && t != CV_32S && t != CV_64S)
                return false;

            if (this->inputs.size() > 1 && !netimpl_->isConstArg(this->inputs[1]))
                return false;

            return true;
        }

        Ptr<BackendNode> initCUDA(void* context_,
                                  InputArrayOfArrays inputs_,
                                  InputArrayOfArrays) CV_OVERRIDE
        {
            auto context = reinterpret_cast<cuda4dnn::csl::CSLContext*>(context_);
            std::vector<UMat> inputsU;
            inputs_.getUMatVector(inputsU);

            int k = 0;
            if (this->inputs.size() > 1) {
                Net::Impl* netimpl_ = getNetImpl(this);
                CV_Assert(netimpl_ && netimpl_->isConstArg(this->inputs[1]));
                Mat kTensor = netimpl_->argTensor(this->inputs[1]).getMat(ACCESS_READ);
                k = static_cast<int>(kTensor.at<int64_t>(0, 0));
            }

            return make_cuda_node_with_type<cuda4dnn::TriluOp>(
                preferableTarget, inputsU[0].type(), std::move(context->stream), k, upperTri);
        }
#endif

        virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                    const int requiredOutputs,
                                    std::vector<MatShape> &outputs,
                                    std::vector<MatShape> &internals) const CV_OVERRIDE {
            outputs.assign(1, inputs[0]);
            return false;
        }

        void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
            std::vector<Mat> inputs, outputs;
            inputs_arr.getMatVector(inputs);
            outputs_arr.getMatVector(outputs);
            inputs[0].copyTo(outputs[0]);
            if (inputs[0].empty())
                return;

            int k = inputs.size() > 1 ? inputs[1].at<int64>(0,0) : 0;
            const auto shape_input = shape(inputs[0]);
            const int cdims = std::max(int(shape_input.size()) - 2, 0);
            const int w = inputs[0].size[shape_input.size() - 1];
            const int h = shape_input.size() >= 2 ? inputs[0].size[shape_input.size() - 2] : 1;

            const int m = std::min(h,w);
            int loops = 1;
            for (int i = 0; i < cdims; ++i)
                loops *= shape_input[i];

            const size_t elemSize = outputs[0].elemSize();
            uchar *dst = outputs[0].ptr<uchar>();
            auto fn = [&](const Range &r) {
                for (int i = r.start; i < r.end; i++) {
                    for(int l=0; l < m; l+=1) {
                        int cmin = upperTri ? 0 : (l + k + 1);
                        cmin = std::max(cmin, 0);
                        const int cmax = upperTri ? min(l + k -1, w-1) : w-1;
                        const int num_zeros = cmax - cmin + 1;
                        size_t offset = (static_cast<size_t>(w) * h * i + (static_cast<size_t>(w) * l + cmin)) * elemSize;
                        auto *cur_dst = dst + offset;
                        if (cmin < w && num_zeros > 0)
                            std::memset(cur_dst, 0, elemSize * num_zeros);
                    }
                }
            };

            double nstripes = loops * h * w / 1024.0;
            parallel_for_(Range(0, loops), fn, nstripes);
        }

        void getTypes(const std::vector<MatType>& inputs,
            const int requiredOutputs,
            const int requiredInternals,
            std::vector<MatType>& outputs,
            std::vector<MatType>& internals) const CV_OVERRIDE
        {
            outputs.assign(1, inputs[0]);
        }

    private:
        bool upperTri;
};

Ptr<TriluLayer> TriluLayer::create(const LayerParams& params)
{
    return makePtr<TriluLayerImpl>(params);
}
}}
