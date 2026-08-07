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
 * \file recv_proto.cc
 * \brief
 */

#include "ops_proto_hccl.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/infer_datatype_context.h"
#include "op_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus HcomReceiveInferShapeV2(gert::InferShapeContext* context)
{
    OP_INFER_SHAPE_START;

    auto attrs = context->GetAttrs();
    OP_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName, "attrs is null"), return GRAPH_FAILED);

    auto shapeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    OP_CHECK(shapeAttr == nullptr, CUBE_INNER_ERR_REPORT(opName, "shapeAttr is null"), return GRAPH_FAILED);

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK(outputShape == nullptr, CUBE_INNER_ERR_REPORT(opName, "output shape is null"), return GRAPH_FAILED);

    auto sizesArray = reinterpret_cast<const int64_t*>(shapeAttr->GetData());
    outputShape->SetDimNum(shapeAttr->GetSize());

    for (size_t i = 0; i < shapeAttr->GetSize(); ++i) {
        if (sizesArray[i] <= 0) {
            OP_LOGE(opName, "value of sizes[%ld] must greater than 0, but got %lu", i, sizesArray[i]);
            return GRAPH_FAILED;
        }
        outputShape->SetDim(i, sizesArray[i]);
    }

    OP_INFER_SHAPE_END;
    return GRAPH_SUCCESS;
}

static ge::graphStatus HcomReceiveInferDataTypeV2(gert::InferDataTypeContext* context)
{
    OP_INFER_DATATYPE_START;

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);

    OP_INFER_DATATYPE_END;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcomReceive).InferShape(HcomReceiveInferShapeV2).InferDataType(HcomReceiveInferDataTypeV2);
} // namespace ops
