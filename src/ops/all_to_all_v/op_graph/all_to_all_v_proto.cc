/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_to_all_v_proto.cc
 * \brief
 */

#include "ops_proto_hccl.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/infer_datatype_context.h"
#include "op_util.h"
#include "base/alog_pub.h"
#include <cmath>

using namespace ge;

namespace ops {

static bool HcomAllToAllShapeFullDefined(const gert::Shape* shape)
{
    if (shape == nullptr) {
        return false;
    }
    for (size_t i = 0; i < shape->GetDimNum(); i++) {
        if (shape->GetDim(i) == ge::UNKNOWN_DIM) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus HcomAllToAllInferShapeV2(gert::InferShapeContext* context)
{
    AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_DEBUG, "[HCCL_PROTO] %s enter.", context->GetNodeName());
    OP_INFER_SHAPE_START;

    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    if (!HcomAllToAllShapeFullDefined(inputShape)) {
        *outputShape = *inputShape;
        OP_LOGI(opName, "the op infershape end, shape is unknown.");
        return GRAPH_SUCCESS;
    }

    if (inputShape->GetDimNum() == 0) {
        CUBE_INNER_ERR_REPORT(opName, "input tensor's first dim is illegal, expected: > 0, actual: 0.");
        return GRAPH_FAILED;
    }

    *outputShape = *inputShape;

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllVInferShapeV2(gert::InferShapeContext* context)
{
    AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_DEBUG, "[HCCL_PROTO] %s enter.", context->GetNodeName());
    OP_INFER_SHAPE_START;

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    const gert::Tensor* recvDispTensor = context->GetInputTensor(4);
    const gert::Tensor* recvCountsTensor = context->GetInputTensor(3);

    if (!HcomIsConstData(opName, recvDispTensor) || !HcomIsConstData(opName, recvCountsTensor)) {
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        OP_LOGI(opName, "[%s] the op inferShape unknown.", __func__);
        return GRAPH_SUCCESS;
    }

    auto recvDispDtype = context->GetInputTensor(4);
    vector<int64_t> recvDisp;
    HcomGetConstValue(opName, recvDispTensor, recvDispDtype->GetDataType(), recvDisp);

    auto recvCountsDtype = context->GetInputTensor(3);
    vector<int64_t> recvCounts;
    HcomGetConstValue(opName, recvCountsTensor, recvCountsDtype->GetDataType(), recvCounts);

    if (recvDisp.size() != recvCounts.size()) {
        OP_LOGE(
            opName, "recvDisp size[%zu] and recvCounts size[%zu] are different.", recvDisp.size(), recvCounts.size());
        return GRAPH_FAILED;
    }

    int64_t recvShape = -1;
    for (size_t i = 0; i < recvDisp.size(); i++) {
        int64_t tempSum = recvDisp[i] + recvCounts[i];
        if (recvShape < tempSum) {
            recvShape = tempSum;
        }
    }

    outputShape->SetDimNum(1);
    outputShape->SetDim(0, recvShape);

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllVInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllVCInferShapeV2(gert::InferShapeContext* context)
{
    AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_DEBUG, "[HCCL_PROTO] %s enter.", context->GetNodeName());
    OP_INFER_SHAPE_START;

    // Get RuntimeAttrs
    auto attrs = context->GetAttrs();
    constexpr size_t a2aFusionIndex = 2;
    constexpr size_t a2aFusionIdIndex = 3;
    if (CheckOPAttr(opName, attrs, a2aFusionIndex, a2aFusionIdIndex) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    const gert::Tensor* sendCountMatrixTensor = context->GetInputTensor(1);

    if (!HcomIsConstData(opName, sendCountMatrixTensor)) {
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        OP_LOGI(opName, "[%s] the op inferShape unknown.", __func__);
        return GRAPH_SUCCESS;
    }

    vector<int64_t> sendCountMatrix;
    auto sendCountDtype = context->GetInputTensor(1);
    HcomGetConstValue(opName, sendCountMatrixTensor, sendCountDtype->GetDataType(), sendCountMatrix);

    for (size_t i = 0; i < sendCountMatrix.size(); ++i) {
        OP_LOGD(opName, "[%s] sendCountMatrix : %zu : %ld ", __func__, i, sendCountMatrix[i]);
    }

    constexpr size_t rankIndex = 0;
    auto rankPtr = attrs->GetAttrPointer<int64_t>(rankIndex);
    OP_CHECK(rankPtr == nullptr, CUBE_INNER_ERR_REPORT(opName, "attr rank is null"), return GRAPH_FAILED);
    int64_t rank = *rankPtr;
    int64_t rankSize = static_cast<int64_t>(sqrt(sendCountMatrix.size()));
    if (rankSize <= 0) {
        OP_LOGE(opName, "rankSize is illegal, expected: > 0, actual: %ld.", rankSize);
        return GRAPH_FAILED;
    }
    if (rank < 0 || rank >= rankSize) {
        OP_LOGE(
            opName,
            "attr rank: %ld is illegal, expected:"
            "[0 ~ %ld]",
            rank, rankSize - 1);
        return GRAPH_FAILED;
    }

    int64_t recvCount = 0;
    for (int64_t i = 0; i < rankSize; i++) {
        int64_t tempRecvCount = sendCountMatrix[rank + i * rankSize];
        recvCount += tempRecvCount;
    }

    OP_LOGD(opName, "[%s] alltoallvc recvCount : %ld. rank: %ld rankSize : %ld ", __func__, recvCount, rank, rankSize);
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, recvCount);

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllToAllVCInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomAllToAll).InferShape(HcomAllToAllInferShapeV2).InferDataType(HcomAllToAllInferDataTypeV2);
IMPL_OP_INFERSHAPE(HcomAllToAllV)
    .InferShape(HcomAllToAllVInferShapeV2)
    .InferDataType(HcomAllToAllVInferDataTypeV2)
    .InputsDataDependency({1, 2, 3, 4});
IMPL_OP_INFERSHAPE(HcomAllToAllVC)
    .InferShape(HcomAllToAllVCInferShapeV2)
    .InferDataType(HcomAllToAllVCInferDataTypeV2)
    .InputsDataDependency({1});
} // namespace ops
