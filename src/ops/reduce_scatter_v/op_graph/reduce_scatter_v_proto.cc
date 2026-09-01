/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_util.h"
#include "base/alog_pub.h"
#include "ops_proto_hccl.h"
#include "runtime/infer_datatype_context.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include <cmath>

using namespace ge;

namespace ops {

static ge::graphStatus HcomReduceScatterVInferShapeV2(gert::InferShapeContext* context)
{
    AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_DEBUG, "[HCCL_PROTO] %s enter.", context->GetNodeName());
    OP_INFER_SHAPE_START;

    auto attrs = context->GetAttrs();
    constexpr size_t reduceScatterVReductionIndex = 0;
    if (CheckReductionAttr(opName, attrs, reduceScatterVReductionIndex, false) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    const gert::Tensor* recvCountTensor = context->GetInputTensor(1);
    const gert::Tensor* sendCountsTensor = context->GetInputTensor(2);

    if (!HcomIsConstData(opName, recvCountTensor) || !HcomIsConstData(opName, sendCountsTensor)) {
        *outputShape = *inputShape;
        OP_LOGI(opName, "the op infershape end, shape first dim is unknown.");
        return GRAPH_SUCCESS;
    }

    auto recvCountDtype = context->GetInputTensor(1);
    std::vector<int64_t> recvCount;
    HcomGetConstValue(opName, recvCountTensor, recvCountDtype->GetDataType(), recvCount);

    if (recvCount.empty()) {
        CUBE_INNER_ERR_REPORT(opName, "recv_count is empty or dtype is not supported.");
        return GRAPH_FAILED;
    }

    if (inputShape->GetDimNum() == 0) {
        CUBE_INNER_ERR_REPORT(opName, "input tensor's first dim is illegal, expected: > 0, actual: 0.");
        return GRAPH_FAILED;
    }

    // recvCount 是元素个数, 需除以 otherDims 得到首维个数
    int64_t otherDims = 1;
    for (size_t i = 1; i < inputShape->GetDimNum(); i++) {
        otherDims *= inputShape->GetDim(i);
    }
    if (otherDims == 0) {
        CUBE_INNER_ERR_REPORT(opName, "otherDims is 0, input shape may contain zero dim.");
        return GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    outputShape->SetDim(0, recvCount[0] / otherDims);

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomReduceScatterVInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomReduceScatterV)
    .InferShape(HcomReduceScatterVInferShapeV2)
    .InferDataType(HcomReduceScatterVInferDataTypeV2)
    .InputsDataDependency({1, 2, 3});
} // namespace ops
