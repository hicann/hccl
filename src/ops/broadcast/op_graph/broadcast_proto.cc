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
 * \file broadcast_proto.cc
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

static ge::graphStatus HcomBroadcastInferShapeV2(gert::InferShapeContext* context)
{
    AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_DEBUG, "[HCCL_PROTO] %s enter.", context->GetNodeName());
    OP_INFER_SHAPE_START;

    // Get RuntimeAttrs
    auto attrs = context->GetAttrs();
    constexpr size_t bcastFusionIndex = 2;
    constexpr size_t bcastFusionIdIndex = 3;
    if (CheckOPAttr(opName, attrs, bcastFusionIndex, bcastFusionIdIndex) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    uint32_t UINT_MAX_VALUE = 0xFFFFFFFF;
    uint32_t inputSize = context->GetComputeNodeInputNum();
    if (inputSize >= UINT_MAX_VALUE) {
        CUBE_INNER_ERR_REPORT(opName, "GetInputSize [%u] is more than %u", inputSize, UINT_MAX_VALUE);
        return GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputSize; i++) {
        const auto inputShape = context->GetInputShape(i);
        OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
        auto outputShape = context->GetOutputShape(i);
        OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);
        *outputShape = *inputShape;
    }

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomBroadcastInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    const unsigned int UINT_MAX_VALUE = 0xFFFFFFFF;
    uint32_t inputSize = context->GetComputeNodeInputNum();
    if (inputSize >= UINT_MAX_VALUE) {
        OP_LOGE(opName, "GetInputSize [%u] is more than %u", inputSize, UINT_MAX_VALUE);
        return GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputSize; i++) {
        ge::DataType inputType = context->GetInputDataType(i);
        context->SetOutputDataType(i, inputType);
    }

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomBroadcast).InferShape(HcomBroadcastInferShapeV2).InferDataType(HcomBroadcastInferDataTypeV2);
} // namespace ops
