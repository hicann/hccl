/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_REDUCE_SCATTER_V_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_REDUCE_SCATTER_V_OP_H

#include "hccl_custom_op.h"

// 不继承 ge::ArgsUpdater (GE dynamic_cast 判定失败 → 不走 args 分配/地址刷新, 符合预期)
class HcomReduceScatterVOp : public hccl::HcclCustomOpBase {
public:
    ~HcomReduceScatterVOp() override = default;
    const char* GetOpTypeName() const override;

    // 输入/属性下标 (与安装头 es_HcomReduceScatterV_c.h 签名对齐)
    static constexpr size_t INPUT_X = 0;
    static constexpr size_t INPUT_RECV_COUNT = 1;
    static constexpr size_t INPUT_SEND_COUNTS = 2;
    static constexpr size_t INPUT_SEND_DISPLS = 3;
    static constexpr size_t ATTR_REDUCTION = 0;
    static constexpr size_t ATTR_GROUP = 1;

protected:
    ge::graphStatus ExtractParamsStatic(hccl::HcclOpState& st) override;
    ge::graphStatus CalcResourcesStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpDynamic(hccl::HcclOpState& st) override; // 纯虚, 必须显式实现
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override;
    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override;

    // 不覆写 ExtractParamsDynamic: RSV 动静态提取逻辑一致 (均需读 counts + MallocOutputTensor),
    // 走基类默认实现 (转调 ExtractParamsStatic)

private:
    // attr 提取 + dataType/reduceOp 转换校验 (reduction/group 为 RSV 的 attr, group 出参生命周期随 ctx)
    ge::graphStatus ExtractVAttrs(hccl::HcclOpState& st, const char*& group);
    // outDim0 = rawRecvCount[0] (recv_count 为首维单位, 与上游 proto 行为一致); uint64 → int64 窄化校验
    ge::graphStatus CheckOutDim0(uint64_t rawOutDim0, int64_t& outDim0);
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_REDUCE_SCATTER_V_OP_H
