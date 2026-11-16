/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AIV_TEMP_BATCH_SEND_RECV_MESH_1D_H
#define HCCLV2_AIV_TEMP_BATCH_SEND_RECV_MESH_1D_H

#include <string>
#include <vector>

#include "aiv_alg_template_base.h"
#include "template_utils.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {

// 本算子把 tag 随 sliceId 下发，kernel 侧算 flag 值 = (tag << 16) | sliceIdx，而 WaitFlag 是精确相等比较，
// 所以 tag 值域必须与其它 AIV 算子隔离：它们等完不清 flag（all_gather 的 RunCtrlCore 就只写不清），
// 残留在同一个 slot 里的旧值一旦与本轮期望值相等，wait 会在数据搬完之前就返回，静默读脏数据。
// 其它算子的 tag 恒在 [TAG_INIT_VALUE, TAG_RESET_COUNT] = [1, 1000]（GetTag 到 1000 回绕），故 BSR 的 tag
// 从 AIV_BSR_TAG_BASE+1 = 1025 起取，两个值域不相交：别人的残留永远不等于我们的期望值，我们的残留也骗不到
// 别的算子。上限 0x7FFF 是 tag<<16 不左移进 int32 符号位的边界，越界按 AIV_BSR_TAG_SPAN 回绕（与其它算子
// 到 1000 回绕同理），回绕后的最小值仍是 1025，不会掉回别人的值域。
constexpr u32 AIV_BSR_TAG_BASE = 1024U;
constexpr u32 AIV_BSR_TAG_MAX = 0x7FFFU;
// 回绕环长：tag 取值 [AIV_BSR_TAG_BASE + 1, AIV_BSR_TAG_MAX] 共 SPAN 个，末值 +1 回绕到 1025。
constexpr u32 AIV_BSR_TAG_SPAN = AIV_BSR_TAG_MAX - AIV_BSR_TAG_BASE;

class AivTempBatchSendRecvMesh1D : public AivAlgTemplateBase {
public:
    AivTempBatchSendRecvMesh1D() = default;
    AivTempBatchSendRecvMesh1D(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~AivTempBatchSendRecvMesh1D() override = default;

    std::string Describe() const override
    {
        std::string info = "Template of BatchSendRecv Mesh1D with tempRankSize ";
        info += std::to_string(tempRankSize_);
        return info;
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;

    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) override;

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams,
        const TemplateResource& templateResource) override;

    // 每个 round 的收发任务描述由 executor 切片后下发
    void SetBatchSendRecvItems(const ExtraArgsBsr& extraArgs) { extraArgs_ = extraArgs; }

    // DFX 上报的本 rank 本轮真正发出的字节数，由 executor 铺任务表时顺便累加后传入（只喂 taskInfo.size）。
    void SetDfxTotalBytes(u64 totalBytes) { dfxTotalBytes_ = totalBytes; }

    // 0 起的轮次序号：KernelRun 里换算成 tag（AIV_BSR_TAG_BASE + 1 + roundIdx，越界回绕）随 sliceId 下发，
    // 同时用于 host 日志区分轮次。
    void SetRoundIdx(u32 roundIdx) { roundIdx_ = roundIdx; }

private:
    ExtraArgsBsr extraArgs_{};
    u64 dfxTotalBytes_ = 0;
    u32 roundIdx_ = 0;
};

} // namespace ops_hccl

#endif // HCCLV2_AIV_TEMP_BATCH_SEND_RECV_MESH_1D_H
