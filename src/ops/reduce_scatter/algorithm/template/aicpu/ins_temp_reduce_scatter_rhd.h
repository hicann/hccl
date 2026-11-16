/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_RHD_H
#define INS_TEMP_REDUCE_SCATTER_RHD_H

#include <map>
#include <vector>
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "alg_data_trans_wrapper.h"
#include "cost_model.h"

namespace ops_hccl {

// ReduceScatter RHD（递归折半倍增，支持非 2 次幂）AICPU 模板（多通道）：
// = 标准 HD 主体 + 非 2 幂前/后处理；步骤表统一调 template_utils RHD 函数族（σ 份额槽位布局）。
// 三段执行：PRE（被合并方→代表，全量段 [0,p) write-reduce 进同号槽）；MAIN（log₂P 步，每步
// tx/rx 整段一条 DataSlice——合并发送禁逐 slice，SendRecvBatchWriteReduce 双向 RMW，无步间
// barrier）；POST（代表份额槽整块覆盖写回被合并方同号槽）。
// scratch 为 p 槽单缓冲：algIdx i 镜像进槽 σ(i)（配对份额相邻、代表持 [2t,2t+2)），交换区段
// 槽位坐标连续是合并发送（每步每对端 1 条 WQE）的前提。通道切分：镜像/回搬按单槽水平切分，
// 交换相位对【合并后大段】比例切分、每通道单条连续子段（BuildSegSlices，需求 5.1.2）。
// 寻址按五参数契约 IN[base+rpt*IRS+i*ISS] / OUT[base+rpt*ORS+myAlgIdx*OSS]；尾块由 executor
// loopTimes 拆到模板外（tailSize==sliceSize）。preferred-only：topoCustomCheck 恒 false +
// 占位 cost 恒非空（空系数在 cost 表注册阶段即被跳过，显式配置也无法选中）。
// 绑定：Sequence 二级 L1 / 三级 L1/L2（Aicpu 前缀），与 Dpu 三级 L1 共用同一实例。
// 复用开关：TemplateDataParams.rhdSkipPost 置位时跳过 POST 与被合并方终态回搬——供
// Reduce/AllReduce 等上层算子复用 RS 相位（被合并方结果由上层承担，需求 5.1.1 post=跳过/FULL）。
class InsTempReduceScatterRHD : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::RHD};
    InsTempReduceScatterRHD() = default;
    explicit InsTempReduceScatterRHD(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempReduceScatterRHD() override;

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter RHD with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult HandleSingleRank(const std::vector<ThreadHandle>& threads);
    HcclResult SingleRankCopy(const std::vector<ThreadHandle>& threads, u64 rpt);
    HcclResult LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx); // p 槽全量镜像（σ 布局）
    HcclResult RunPre(const std::vector<ThreadHandle>& threads, u32 channelIdx);        // 前处理（仅非 2 幂）
    HcclResult RunMain(const std::vector<ThreadHandle>& threads, u32 channelIdx);       // 主体（合并发送）
    HcclResult RunPost(const std::vector<ThreadHandle>& threads, u32 channelIdx);       // 后处理（仅非 2 幂）
    HcclResult PostLocalCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx); // 份额槽终态回搬
    u64 RptScratchBase(u64 rpt) const;
    // σ unit 变长前缀偏移（[检视预留 AllReduce 复用] tailSize != sliceSize 时末 unit 更大）；
    // 均匀退化（tailSize == sliceSize）时 unitOffsets_[u] == u*sliceSize，与原行为逐字节一致
    std::vector<u64> unitOffsets_;
    u64 totalUnitBytes_{0}; // Σ unit 尺寸（= 每 repeat scratch 跨度；均匀时 = p*SS）
    u64 tailLen_{0};        // 末 unit 实际尺寸（executor 恒填 tailSize，防御 0 时取 sliceSize）
    HcclResult BuildUnitOffsets();
    HcclResult BuildSegSlices(
        void* addr, u64 base, u32 segStart, u32 segSlots, u32 channelIdx,
        std::vector<DataSlice>& slices) const; // 槽位段切片（合并发送：段级通道切分）

    TemplateDataParams tempAlgParams_;
    std::map<u32, std::vector<ChannelInfo>> channels_;
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;
    RhdPart1Info part1Info_{};
    std::vector<RhdStepInfo> rsSteps_;
    RhdStepInfo preStep_{};
    RhdStepInfo postStep_{};
    u32 threadNum_{0};
    u64 dataTypeSize_{0};
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_RHD_H
