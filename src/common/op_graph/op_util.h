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

#include <string>
#include "log.h"
#include "runtime/tensor.h"
#include "runtime/runtime_attrs.h"
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

inline bool HcomIsConstData(const ge::char_t* opName, const gert::Tensor* tensor)
{
    if (tensor == nullptr) {
        OP_LOGE(opName, "the op shape tensor is null.");
        return false;
    }
    return IsConstTensor(tensor);
}

inline void HcomGetConstValue(
    const ge::char_t* opName, const gert::Tensor* const_tensor, const ge::DataType& dtype,
    std::vector<int64_t>& const_data)
{
    if (dtype == ge::DT_INT64) {
        const int64_t* const_data_ptr = const_tensor->GetData<int64_t>();
        size_t size = const_tensor->GetShapeSize();
        OP_LOGD(opName, "size : %zu", size);
        for (size_t i = 0; i < size; ++i) {
            const_data.push_back(*(const_data_ptr + i));
            OP_LOGD(opName, "const data int64 %ld", (int64_t)(*(const_data_ptr + i)));
        }
    } else if (dtype == ge::DT_INT32) {
        const int32_t* const_data_ptr = const_tensor->GetData<int32_t>();
        size_t size = const_tensor->GetShapeSize();
        for (size_t i = 0; i < size; ++i) {
            const_data.push_back(*(const_data_ptr + i));
            OP_LOGD(opName, "const data int32 %d", (int32_t)(*(const_data_ptr + i)));
        }
    }
    return;
}

inline void HcomGetConstValue(
    const ge::char_t* opName, const gert::Tensor* const_tensor, const ge::DataType& dtype,
    std::vector<uint64_t>& const_data)
{
    if (dtype == ge::DT_UINT64) {
        const uint64_t* const_data_ptr = const_tensor->GetData<uint64_t>();
        size_t size = const_tensor->GetShapeSize();
        for (size_t i = 0; i < size; ++i) {
            const_data.push_back(*(const_data_ptr + i));
            OP_LOGD(opName, "const data uint64  %lu", (uint64_t)(*(const_data_ptr + i)));
        }
    }
    return;
}

inline ge::graphStatus CheckFusionAttr(
    const ge::char_t* opName, const gert::RuntimeAttrs* attrs, size_t fusionIndex, size_t fusionIdIndex,
    bool allowFusionAttrOne)
{
    OP_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName, "attrs is null"), return ge::GRAPH_FAILED);

    constexpr int64_t fusionAttrNoFuse = 0;
    constexpr int64_t fusionAttrFuse = 1;
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

    if (allowFusionAttrOne) {
        if ((fusionAttr < fusionAttrNoFuse) || (fusionAttr > fusionAttrFuseById)) {
            OP_LOGE(
                opName, "Attr fusion [%ld] is not supported. expected: [%ld ~ %ld]", fusionAttr, fusionAttrNoFuse,
                fusionAttrFuseById);
            return ge::GRAPH_FAILED;
        }
    } else {
        if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
            OP_LOGE(
                opName, "Attr fusion [%ld] is not supported. expected: [%ld or %ld]", fusionAttr, fusionAttrNoFuse,
                fusionAttrFuseById);
            return ge::GRAPH_FAILED;
        }
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

inline ge::graphStatus
CheckOPAttr(const ge::char_t* opName, const gert::RuntimeAttrs* attrs, size_t fusionIndex, size_t fusionIdIndex)
{
    return CheckFusionAttr(opName, attrs, fusionIndex, fusionIdIndex, false);
}

inline ge::graphStatus
CheckReductionAttr(const ge::char_t* opName, const gert::RuntimeAttrs* attrs, size_t reductionIndex, bool supportProd)
{
    OP_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(opName, "attrs is null"), return ge::GRAPH_FAILED);

    auto reductionPtr = attrs->GetAttrPointer<ge::char_t>(reductionIndex);
    OP_CHECK(reductionPtr == nullptr, CUBE_INNER_ERR_REPORT(opName, "attr reduction is null"), return ge::GRAPH_FAILED);

    std::string reduction(reductionPtr);
    bool valid = (reduction == "min") || (reduction == "max") || (reduction == "sum");
    if (supportProd) {
        valid = valid || (reduction == "prod");
    }
    if (!valid) {
        if (supportProd) {
            OP_LOGE(opName, "Attr reduction [%s] is not supported. expected: min, max, prod, sum", reduction.c_str());
        } else {
            OP_LOGE(opName, "Attr reduction [%s] is not supported. expected: min, max, sum", reduction.c_str());
        }
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckAllReduceFusionAttr(
    const ge::char_t* opName, const gert::RuntimeAttrs* attrs, size_t fusionIndex, size_t fusionIdIndex)
{
    return CheckFusionAttr(opName, attrs, fusionIndex, fusionIdIndex, true);
}

} // namespace ops
#endif // CANN_OPS_BUILT_IN_OP_UTIL_H_
