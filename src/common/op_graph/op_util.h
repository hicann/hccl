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
 * \file op_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_UTIL_H_

#include "log.h"
#include "runtime/tensor.h"
#include "graph/ge_error_codes.h"

namespace ops {

inline const char* get_op_info(const char* str) { return (str == nullptr) ? "nil" : str; }

#define OP_CHECK(cond, log_func, return_expr) \
    if (cond) {                               \
        log_func;                             \
        return_expr;                          \
    }

#define OP_LOGD(opname, format, ...) HCCL_DEBUG("OpName:[%s] " format, get_op_info(opname), ##__VA_ARGS__)

#define OP_LOGI(opname, format, ...) HCCL_INFO("OpName:[%s] " format, get_op_info(opname), ##__VA_ARGS__)

#define OP_LOGW(opname, format, ...) HCCL_WARNING("OpName:[%s] " format, get_op_info(opname), ##__VA_ARGS__)

#define OP_LOGE(opname, format, ...) HCCL_ERROR("OpName:[%s] " format, get_op_info(opname), ##__VA_ARGS__)

#define CUBE_INNER_ERR_REPORT(opname, err_msg, ...) \
    HCCL_ERROR("OpName:[%s] " err_msg, get_op_info(opname), ##__VA_ARGS__)

#define OP_INFER_SHAPE_START                                                                                      \
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED); \
    const auto opName = context->GetNodeName();                                                                   \
    OP_LOGI(opName, "[%s] the op inferShape start.", __func__)

#define OP_INFER_SHAPE_END OP_LOGI(opName, "[%s] the op inferShape end.", __func__)

#define OP_INFER_DATATYPE_START                                                                                   \
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED); \
    const auto opName = context->GetNodeName();                                                                   \
    OP_LOGI(opName, "[%s] the op inferDataType start.", __func__)

#define OP_INFER_DATATYPE_END OP_LOGI(opName, "[%s] the op inferDataType end.", __func__)

inline bool IsConstTensor(const gert::Tensor* input_tensor)
{
    if (input_tensor != nullptr) {
        if (input_tensor->GetAddr() == nullptr) {
            // empty tensor
            return input_tensor->GetShapeSize() == 0;
        }
        return true;
    }
    return false;
}

inline ge::graphStatus
CheckOPAttr(const ge::char_t* opName, const gert::RuntimeAttrs* attrs, size_t fusionIndex, size_t fusionIdIndex)
{
    OP_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName, "attrs is null"), return ge::GRAPH_FAILED);

    constexpr int64_t fusionAttrNoFuse = 0;
    constexpr int64_t fusionAttrFuseById = 2;
    constexpr int64_t fusionIdDefaultVal = -1;
    constexpr int64_t fusionIdMinVal = 0;
    constexpr int64_t fusionIdMaxVal = 0x7fffffff;

    int64_t fusionAttr = fusionAttrNoFuse;
    int64_t fusionIdAttr = fusionIdDefaultVal;

    if ((attrs->GetAttrPointer<int64_t>(fusionIndex)) != nullptr) {
        fusionAttr = *((attrs->GetAttrPointer<int64_t>(fusionIndex)));
    }
    if (attrs->GetAttrPointer<int64_t>(fusionIdIndex) != nullptr) {
        fusionIdAttr = *(attrs->GetAttrPointer<int64_t>(fusionIdIndex));
    }

    if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
        OP_LOGE(
            opName, "Attr fusion [%ld] is not supported. expected: [%ld or %ld]", fusionAttr, fusionAttrNoFuse,
            fusionAttrFuseById);
        return ge::GRAPH_FAILED;
    }
    if (fusionAttr == fusionAttrFuseById) {
        if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
            OP_LOGE(
                opName,
                "In fusion [%ld], attr fusion_id [%ld] is not supported, "
                "expected: [%ld ~ %ld]",
                fusionAttr, fusionIdAttr, fusionIdMinVal, fusionIdMaxVal);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace ops
#endif // CANN_OPS_BUILT_IN_OP_UTIL_H_
