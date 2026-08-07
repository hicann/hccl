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
 * \file all_gather_proto.cc
 * \brief
 */

#include "ops_proto_hccl.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/infer_datatype_context.h"
#include "op_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus HcomAllGatherInferShapeV2(gert::InferShapeContext* context)
{
    OP_INFER_SHAPE_START;

    // Get RuntimeAttrs
    auto attrs = context->GetAttrs();
    constexpr size_t allGatherFusionIndex = 2;
    constexpr size_t allGatherFusionIdIndex = 3;
    if (CheckOPAttr(opName, attrs, allGatherFusionIndex, allGatherFusionIdIndex) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    const auto inputShape = context->GetInputShape(0);
    OP_CHECK(inputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "input shape is null"), return GRAPH_FAILED);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    constexpr size_t rankIndex = 0;
    auto rankSizePtr = attrs->GetAttrPointer<int64_t>(rankIndex);
    OP_CHECK(rankSizePtr == nullptr, CUBE_INNER_ERR_REPORT(opName, "attr rank_size is null"), return GRAPH_FAILED);
    int64_t rankSize = *rankSizePtr;
    OP_CHECK(
        (rankSize <= 0),
        CUBE_INNER_ERR_REPORT(opName, "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize),
        return GRAPH_FAILED);
    // not ShapeFirstDimDefined
    if (inputShape->GetDimNum() > 0 && inputShape->GetDim(0) == ge::UNKNOWN_DIM) {
        *outputShape = *inputShape;
        OP_LOGI(opName, "the op infershape end, shape first dim is unknown.");
        return GRAPH_SUCCESS;
    }
    *outputShape = *inputShape;
    outputShape->SetDim(0, inputShape->GetDim(0) * rankSize);

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomAllGatherInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomAllGather).InferShape(HcomAllGatherInferShapeV2).InferDataType(HcomAllGatherInferDataTypeV2);

} // namespace ops
