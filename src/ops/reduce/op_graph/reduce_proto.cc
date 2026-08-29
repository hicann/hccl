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
 * \file reduce_proto.cc
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

static ge::graphStatus HcomReduceInferShapeV2(gert::InferShapeContext* context)
{
    AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_DEBUG, "[HCCL_PROTO] %s enter.", context->GetNodeName());
    OP_INFER_SHAPE_START;

    auto attrs = context->GetAttrs();
    constexpr size_t reduceReductionIndex = 1;
    if (CheckReductionAttr(opName, attrs, reduceReductionIndex, true) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }
    constexpr size_t reduceFusionIndex = 3;
    constexpr size_t reduceFusionIdIndex = 4;
    if (CheckOPAttr(opName, attrs, reduceFusionIndex, reduceFusionIdIndex) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    uint32_t inputSize = context->GetComputeNodeInputNum();
    OP_LOGD(opName, "[%s] the op inputSize %u ", __func__, inputSize);

    for (uint32_t index = 0; index < inputSize; index++) {
        const auto inputShape = context->GetInputShape(index);
        OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
        auto outputShape = context->GetOutputShape(index);
        OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);
        *outputShape = *inputShape;
    }

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomReduceInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    uint32_t inputSize = context->GetComputeNodeInputNum();

    OP_LOGD(opName, "[%s] the op inputSize %u ", __func__, inputSize);
    for (uint32_t index = 0; index < inputSize; index++) {
        ge::DataType inputType = context->GetInputDataType(index);
        context->SetOutputDataType(index, inputType);
    }

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomReduce).InferShape(HcomReduceInferShapeV2).InferDataType(HcomReduceInferDataTypeV2);
} // namespace ops
