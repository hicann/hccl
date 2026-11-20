/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_SEND_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_SEND_OP_H

#include "acl/acl_rt.h"
#include "hccl_custom_op.h"
#include <cstdint>
#include <vector>

// Send 刷新路径 args struct: MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步。
// D2D ADDR task 用字段 device 地址做二级指针间接寻址 (同 AllReduce HcomAllReduceArgs 机制)。
// Send 单向: 仅 input 侧两字段, 无 output 字段 (InfoStore HcomSendLoop 仅 ② 无 ⑤)。
struct HcomSendArgs {
    void* cclInputPtr; // CCL input buf device 地址 (Execute 写入, 不变)
    void* inputAddr;   // 用户 input device 地址 (UpdateHostArgs 刷新)
};

class HcomSendOp : public hccl::HcclCustomOpBase, public ge::ArgsUpdater {
public:
    ~HcomSendOp() override;
    const char* GetOpTypeName() const override;

    static constexpr size_t INPUT_INDEX = 0;
    // attr 顺序依据 HcomSend IR 原型: group(0) sr_tag(1) dest_rank(2); 无 OUTPUT 锚点
    static constexpr size_t ATTR_GROUP = 0;
    static constexpr size_t ATTR_SR_TAG = 1;
    static constexpr size_t ATTR_DEST_RANK = 2;

    // 静态 count 为精确逻辑字节数(无对齐): 静态 TensorData 按 shape 逻辑尺寸上报, 与 Recv 侧严格
    // 对称。动态 count 保留 32B 对齐(GE lowering 同源)。
    static constexpr uint64_t ALIGN_SIZE_DYNAMIC = 32;

protected:
    // 静态路径 (Direct / Refresh)
    ge::graphStatus ExtractParamsStatic(hccl::HcclOpState& st) override;
    ge::graphStatus CalcResourcesStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpStatic(hccl::HcclOpState& st) override;
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override;
    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override;

    // 动态路径 (两段发送: shape 元数据 + 数据)
    ge::graphStatus ExtractParamsDynamic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpDynamic(hccl::HcclOpState& st) override;

    // ArgsUpdater: sink 模型每次执行前刷新 inputAddr (只刷 input 侧)
    ge::graphStatus UpdateHostArgs(gert::UpdateArgsContext* ctx) override;

private:
    ge::graphStatus GenerateTagVec(hccl::HcclOpState& st);
    ge::graphStatus LaunchDirect(hccl::HcclOpState& st);
    ge::graphStatus LaunchRefresh(hccl::HcclOpState& st);

    // 超长函数拆分 (行为不变, 步骤与日志严格保序): 动态两段发送的第一段
    // shape 元数据组包 (纯函数) 与 元数据发送 (malloc + H2D + HcclSend + 同步)
    static std::vector<int64_t> PackShapeMetadata(const gert::Shape& inputShape);
    static ge::graphStatus SendShapeMetadata(const hccl::HcclOpState& st, const std::vector<int64_t>& sendInfo);
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_SEND_OP_H
