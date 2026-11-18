/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_TO_ALL_HIER_FIRST_STAGE_H_
#define HCCL_CCU_TEMP_ALL_TO_ALL_HIER_FIRST_STAGE_H_

#include <array>
#include "utils.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_temp_all_to_all_hier_stage_base.h"

namespace ops_hccl {

// LaunchKernels 的地址/步长参数
struct LaunchSliceDesc {
    u64 inputAddr = 0;
    u64 cclBuffAddr = 0;
    u64 token = 0;
    u64 blockStride = 0;
    u64 slotStride = 0;
};

/*
 * 二级序列 AllToAll 首 Stage（框间代表组）：INPUT → HCCL_BUFFER。2+6 multilink 时按 dieId 分桶为
 * MAJOR/MINOR 双 kernel 按 dieSplitRatio 分段，其余单 kernel 全量；本地份由各 kernel 按自身分段承担。
 */
class CcuTempAlltoAllHierFirstStage : public CcuTempAlltoAllHierStageBase {
public:
    CcuTempAlltoAllHierFirstStage() = default;
    CcuTempAlltoAllHierFirstStage(
        const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex);
    ~CcuTempAlltoAllHierFirstStage() override = default;

    std::string Describe() const override
    {
        return "Hierarchical AllToAll first stage (inter-board), stageIndex=" + std::to_string(stageIndex_)
               + ", rankSize=" + std::to_string(templateRankSize_);
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    std::string GetAlgoType() const override { return "FirstMeshStage"; }

private:
    // 框间通道划分
    HcclResult PartitionInterBoardChannels(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        std::vector<HcclChannelDesc>& allChannels);
    // 2+6：双 die 桶 → CLOS_MAJOR/CLOS_MINOR 双 kernel 通道（带宽序 + 契约/组序校验 + 分段比例）
    HcclResult
    BuildDualKernelChannels(HcclComm comm, const std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie);
    // 非 2+6：单 kernel 全量，按组序装配
    HcclResult BuildSingleKernelChannels(const std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);
    // 2+6 分段：MAJOR 承载前 majorSize 段、MINOR 承载剩余，边界夹紧后按 128B 向下对齐
    HcclResult CalcSegSize(u32 kernelIdx, u64 fullSliceSize, u64& segSize, u64& segOffset);
    // 下发循环：taskArgs 组装（13 槽，布局见 FastLaunch 槽位常量）→ launch → submitInfo 缓存
    HcclResult LaunchKernels(
        u32 kernelCount, const LaunchSliceDesc& desc, const LoopGroupConfig& config,
        const TemplateDataParams& tempAlgParams, TemplateResource& templateResource);

    bool is2Plus6_ = false;
    uint32_t kernelCount_ = 1;                                   // 2+6 时为 2（CLOS_MAJOR/CLOS_MINOR），否则 1
    double dieSplitRatio_ = 1.0;                                 // MAJOR 承载比例 = majorBw/(majorBw+minorBw)
    uint32_t peerSliceNum_ = 0;                                  // 每对端承载的块数（板内 rank 数 d0）
    std::array<std::vector<HcclChannelDesc>, 2> kernelChannels_; // [0]=CLOS_MAJOR, [1]=CLOS_MINOR
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_TO_ALL_HIER_FIRST_STAGE_H_
