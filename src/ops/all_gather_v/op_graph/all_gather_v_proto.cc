/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_v_proto.cc
 * \brief
 */

#include "ops_proto_hccl.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/infer_datatype_context.h"
#include "op_util.h"
#include "base/alog_pub.h"

using namespace ge;

namespace ops {

static ge::graphStatus HcomAllGatherVInferShapeV2(gert::InferShapeContext* context)
{
    OP_INFER_SHAPE_START;
    AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_DEBUG, "[HCCL_PROTO] %s enter.", opName);

    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    if (inputShape->GetDimNum() == 0) {
        CUBE_INNER_ERR_REPORT(opName, "input tensor's first dim is illegal, expected: > 0, actual: 0.");
        return GRAPH_FAILED;
    }

    const gert::Tensor* recvCountsTensor = context->GetInputTensor(2);
    const gert::Tensor* sendCountTensor = context->GetInputTensor(1);

    if (!HcomIsConstData(opName, recvCountsTensor) || !HcomIsConstData(opName, sendCountTensor)) {
        *outputShape = *inputShape;
        outputShape->SetDim(0, ge::UNKNOWN_DIM);
        OP_LOGI(opName, "the op infershape end, shape first dim is unknown.");
        return GRAPH_SUCCESS;
    }

    vector<int64_t> recvCounts;
    HcomGetConstValue(opName, recvCountsTensor, recvCountsTensor->GetDataType(), recvCounts);
    if (recvCounts.empty()) {
        CUBE_INNER_ERR_REPORT(opName, "recv_counts is empty or dtype is not supported.");
        return GRAPH_FAILED;
    }

    // 计算recvDisp：recv_displacements为可选输入(index 3)，有const数据时使用真实值，
    // 否则按recvCounts连续累加（与HcomVInputVecSetPass中CalcDisplacements的行为一致）
    const gert::Tensor* recvDisplsTensor = context->GetInputTensor(3);
    vector<int64_t> recvDisp;
    if ((recvDisplsTensor != nullptr) && HcomIsConstData(opName, recvDisplsTensor)) {
        HcomGetConstValue(opName, recvDisplsTensor, recvDisplsTensor->GetDataType(), recvDisp);
        if (recvDisp.size() != recvCounts.size()) {
            CUBE_INNER_ERR_REPORT(
                opName, "recv_displacements size[%zu] must equal recv_counts size[%zu].", recvDisp.size(),
                recvCounts.size());
            return GRAPH_FAILED;
        }
    } else {
        int64_t tempSum = 0;
        for (size_t i = 0; i < recvCounts.size(); i++) {
            recvDisp.push_back(tempSum);
            tempSum += recvCounts[i];
        }
    }

    // recv_counts 的单位为首维长度，输出首维 = max(recvDisp[i] + recvCounts[i])
    int64_t outDim = 0;
    for (size_t i = 0; i < recvCounts.size(); i++) {
        int64_t tempRecvSum = recvDisp[i] + recvCounts[i];
        if (outDim < tempRecvSum) {
            outDim = tempRecvSum;
        }
    }

    *outputShape = *inputShape;
    outputShape->SetDim(0, outDim);

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllGatherVInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomAllGatherV)
    .InferShape(HcomAllGatherVInferShapeV2)
    .InferDataType(HcomAllGatherVInferDataTypeV2)
    .InputsDataDependency({1, 2, 3});
} // namespace ops
