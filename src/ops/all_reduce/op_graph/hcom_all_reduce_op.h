/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H

#include "acl/acl_rt.h"
#include "hccl_custom_op.h"
#include "hccl_op_utils.h"
#include <cstdint>

// 地址刷新 args struct: MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步
// D2D ADDR task 用字段 device 地址做二级指针间接寻址
struct HcomAllReduceArgs {
    void* cclInputPtr;  // CCL input buf device 地址 (Execute 写入, 不变)
    void* cclOutputPtr; // CCL output buf device 地址 (Execute 写入, 不变)
    void* inputAddr;    // 用户 input device 地址 (UpdateHostArgs 刷新)
    void* outputAddr;   // 用户 output device 地址 (UpdateHostArgs 刷新)
};

class HcomAllReduceOp : public hccl::HcclCustomOpBase, public ge::ArgsUpdater {
public:
    const char* GetOpTypeName() const override;
    static constexpr size_t INPUT_INDEX = 0;
    static constexpr size_t OUTPUT_INDEX = 0;
    static constexpr size_t ATTR_REDUCTION = 0;
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
    // 静态路径 (CalcOnlineResources/GenerateTagVec 的公共实现见 hccl_op_utils)
    ge::graphStatus LaunchDirect(hccl::HcclOpState& st);
    ge::graphStatus LaunchRefresh(hccl::HcclOpState& st);
    ge::graphStatus GenerateTagVec(hccl::HcclOpState& st);

    // 动态路径
    ge::graphStatus LaunchBatchDynamic(hccl::HcclOpState& st);
    ge::graphStatus
    CalcBatchSlices(const hccl::HcclOpState& st, uint64_t maxCommCount, std::vector<uint32_t>& sliceIdxs);
    ge::graphStatus CopyInputsToCCLbuff(
        const hccl::HcclOpState& st, uint32_t inputOffset, uint32_t inputsNum, void* cclBuff, uint64_t cclBuffSize,
        uint64_t& commCount);
    ge::graphStatus
    CopyCCLbuffToOutputs(const hccl::HcclOpState& st, uint32_t outputOffset, uint32_t outputsNum, void* cclBuff);
    ge::graphStatus ExecuteBatchSlices(
        hccl::HcclOpState& st, const std::vector<uint32_t>& sliceIdxs, void* commInputPtr, uint64_t commInputSize,
        void* commOutputPtr);

    // crack 零缓冲清理器 (无状态, 缓冲经 MallocWorkSpace 按次申请, 见 hccl_op_utils.h)
    hccl::CrackCleaner crackCleaner_;
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_ALL_REDUCE_OP_H
