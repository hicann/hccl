/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_BROADCAST_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_BROADCAST_OP_H

#include "acl/acl_rt.h"
#include "hccl_custom_op.h"
#include <cstdint>

// 地址刷新 args struct: MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步
// D2D ADDR task 用字段 device 地址做二级指针间接寻址
// Broadcast 是单 buf in-place 且无膨胀 (output = input, 无 cclOutputPtr), 仅 Refresh 路径使用
// 3 字段 (区别于 AG 的 4 字段); Broadcast 在 InCCLbuf 上原地执行, 结果留在原处
struct HcomBroadcastArgs {
    void* cclInputPtr; // CCL input buf device 地址 (Execute 写入, 不变; Broadcast 在其上原地执行)
    void* inputAddr;   // 用户 input device 地址 (UpdateHostArgs 刷新)
    void* outputAddr;  // 用户 output device 地址 (UpdateHostArgs 刷新; 经 MakeOutputRefInput 与 input 同址)
};

class HcomBroadcastOp : public hccl::HcclCustomOpBase, public ge::ArgsUpdater {
public:
    const char* GetOpTypeName() const override;
    static constexpr size_t INPUT_INDEX = 0;
    static constexpr size_t OUTPUT_INDEX = 0;
    // attrs 索引对照本仓 ops_proto_hccl.h 的 HcomBroadcast IR 定义顺序:
    // REQUIRED_ATTR(root_rank, Int) / REQUIRED_ATTR(group, String) / ATTR(fusion) / ATTR(fusion_id)
    // 区别于 AG: index 0 是 root_rank (非 rank_size); 无 reduction 属性
    static constexpr size_t ATTR_ROOT_RANK = 0;
    static constexpr size_t ATTR_GROUP = 1;
    static constexpr size_t ATTR_FUSION = 2;
    static constexpr size_t ATTR_FUSION_ID = 3;

protected:
    ge::graphStatus ExtractParamsStatic(hccl::HcclOpState& st) override;
    ge::graphStatus CalcResourcesStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpStatic(hccl::HcclOpState& st) override;
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override;
    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override;

    ge::graphStatus ExtractParamsDynamic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpDynamic(hccl::HcclOpState& st) override;
    ge::graphStatus UpdateHostArgs(gert::UpdateArgsContext* ctx) override;

private:
    // 静态路径 (CalcOnlineResources/BuildTagVec 的公共实现见 hccl_op_utils)
    ge::graphStatus LaunchDirect(hccl::HcclOpState& st);
    ge::graphStatus LaunchRefresh(hccl::HcclOpState& st);
    ge::graphStatus GenerateTagVec(hccl::HcclOpState& st);

    // Broadcast 无 crack 清零 (InfoStore GetTensorNum 清零列表仅 RS/AllReduce/Reduce)
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_BROADCAST_OP_H
