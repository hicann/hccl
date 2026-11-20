/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_RECEIVE_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_RECEIVE_OP_H

#include "acl/acl_rt.h"
#include "hccl_custom_op.h"
#include <cstdint>
#include <vector>

// Receive 刷新路径 args struct (单向 output, 无 input 字段; InfoStore HcomReceiveLoop 先通信后 ⑤)。
struct HcomRecvArgs {
    void* cclOutputPtr; // CCL output buf device 地址 (Execute 写入, 不变)
    void* outputAddr;   // 用户 output device 地址 (UpdateHostArgs 刷新)
};

// Receive 算子 (HcomReceive): P2P 接收, 无输入张量。
// 静态: output shape 由节点 shape attr 决定; 动态: Execute 内运行时 shape 协商 (对端 Send 先发元数据再发数据)。
// 线协议常量 hccl::SHAPE_INFO_COUNT 定义于 hccl_op_utils.h。

class HcomReceiveOp : public hccl::HcclCustomOpBase, public ge::ArgsUpdater {
public:
    ~HcomReceiveOp() override;
    const char* GetOpTypeName() const override;

    static constexpr size_t OUTPUT_INDEX = 0;
    // attr 顺序依据 HcomReceive IR 原型: group(0) sr_tag(1) src_rank(2) shape(3) dtype(4)。
    // eager 模式 RuntimeAttrs 按 IR 原型 index 读取 (与 AllReduce 的 ATTR_* 常量同机制)。
    static constexpr size_t ATTR_GROUP = 0;
    static constexpr size_t ATTR_SR_TAG = 1;
    static constexpr size_t ATTR_SRC_RANK = 2;
    static constexpr size_t ATTR_SHAPE = 3;
    static constexpr size_t ATTR_DTYPE = 4;

    // 静态 count 为精确逻辑字节数(无对齐): 静态 TensorData 按 shape 逻辑尺寸上报, 对齐口径仅旧
    // InfoStore 路径适用(其分配同步 512 对齐)。动态 count 保留 32B 对齐(与 Send 侧严格对称)。
    static constexpr uint64_t ALIGN_SIZE_DYNAMIC = 32;

protected:
    // 静态路径 (Direct / Refresh)
    ge::graphStatus ExtractParamsStatic(hccl::HcclOpState& st) override;
    ge::graphStatus CalcResourcesStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpStatic(hccl::HcclOpState& st) override;
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override;
    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override;

    // 动态路径: 运行时 shape 协商 (对照 LaunchRecvKernelV2)
    ge::graphStatus ExtractParamsDynamic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpDynamic(hccl::HcclOpState& st) override;

    // ArgsUpdater: sink 模型每次执行前刷新 outputAddr (只刷 output 侧; 无输入节点 GetInputTensor 恒 nullptr)
    ge::graphStatus UpdateHostArgs(gert::UpdateArgsContext* ctx) override;

private:
    ge::graphStatus GenerateTagVec(hccl::HcclOpState& st);
    ge::graphStatus LaunchDirect(hccl::HcclOpState& st);
    ge::graphStatus LaunchRefresh(hccl::HcclOpState& st);

    // 超长函数拆分 (行为不变, 步骤与日志严格保序):
    // 静态路径: output 分配段 (MallocOutputTensor + 分配粒度防御)
    static ge::graphStatus
    AllocateStaticOutput(hccl::HcclOpState& st, const gert::Shape& outputShape, int64_t dtype, size_t& outputSize);
    // 动态路径: 元数据接收 / 协商 shape 解析 / output 分配 (对照 HcomGetRecvBeforeKernelV2 →
    // LaunchRecvKernelV2 的三步)
    static ge::graphStatus ReceiveShapeMetadata(const hccl::HcclOpState& st, std::vector<int64_t>& recvShape);
    static ge::graphStatus ParseNegotiatedShape(
        const std::vector<int64_t>& recvShape, uint64_t unitSize, gert::Shape& negotiatedShape, int64_t& shapeSize,
        uint64_t& dataSize, uint64_t& count);
    static ge::graphStatus
    AllocateNegotiatedOutput(hccl::HcclOpState& st, const gert::Shape& negotiatedShape, size_t& outputSize);
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_RECEIVE_OP_H
