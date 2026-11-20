/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_ALL_GATHER_V_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_ALL_GATHER_V_OP_H

#include "hccl_custom_op.h"

// 不继承 ge::ArgsUpdater (GE dynamic_cast 判定失败 → 不走 args 分配/地址刷新, 符合预期)
class HcomAllGatherVOp : public hccl::HcclCustomOpBase {
public:
    ~HcomAllGatherVOp() override = default;
    const char* GetOpTypeName() const override;

    // 输入/属性下标 (与安装头 es_HcomAllGatherV_c.h 签名对齐)
    static constexpr size_t INPUT_X = 0;
    static constexpr size_t INPUT_SEND_COUNT = 1;
    static constexpr size_t INPUT_RECV_COUNTS = 2;
    static constexpr size_t INPUT_RECV_DISPLS = 3;
    static constexpr size_t ATTR_GROUP = 0; // AGV 无 reduction attr, group 是唯一属性

protected:
    ge::graphStatus ExtractParamsStatic(hccl::HcclOpState& st) override;
    ge::graphStatus CalcResourcesStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpDynamic(hccl::HcclOpState& st) override; // 纯虚, 必须显式实现
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override;
    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override;

    // 不覆写 ExtractParamsDynamic: AGV 动静态提取逻辑一致 (均需读 counts + MallocOutputTensor),
    // 走基类默认实现 (转调 ExtractParamsStatic)

private:
    // attr 提取 + dataType 转换校验 (AGV 无 reduction attr, group 是唯一属性, 出参生命周期随 ctx)
    ge::graphStatus ExtractVAttrs(hccl::HcclOpState& st, const char*& group);
    // outDim0 = max(dimDispls[i] + rawRecvCounts[i]) (首维单位), 含 uint64 加法溢出与 int64 窄化校验
    ge::graphStatus
    CalcOutDim0(const std::vector<uint64_t>& dimDispls, const std::vector<uint64_t>& rawRecvCounts, int64_t& outDim0);
    // sendCount = send_count[0] × otherDims, 含 uint64 乘法溢出校验 (与 CalcOutDim0 的 fail-fast 口径一致)
    ge::graphStatus CalcSendCount(uint64_t rawSendCount, uint64_t otherDims, uint64_t& sendCount);
    // InferShape 的 recvDispls 解析: 非空 const 位移须与 recv_counts 等长, 否则按前缀和
    // (对齐上游 proto 与 vinputvec pass CalcDisplacements)
    ge::graphStatus ResolveRecvDisplsForInferShape(
        gert::InferShapeContext* ctx, const std::vector<uint64_t>& recvCounts, std::vector<uint64_t>& recvDispls);
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_ALL_GATHER_V_OP_H
