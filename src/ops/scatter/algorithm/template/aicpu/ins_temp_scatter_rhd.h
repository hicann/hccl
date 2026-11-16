/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_RHD_H
#define INS_TEMP_SCATTER_RHD_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "cost_model.h"
#include "rhd_scatter_step.h"

namespace ops_hccl {

// Scatter RHD 设备侧模板（AICPU 二级 L1 / 三级 L1、L2；DPU 三级 L1 复用同一实例）。
// 步骤生成消费 rhd_scatter_step 骨架（GenRhdScatterSteps），区段字节换算消费公共函数
// template_utils（RhdSegToBytesUniform / RhdUnitStartOfRank），模板内不私有实现步骤公式。
// 合并发送硬约束：每步对每个对端只构造单条连续大块 DataSlice，禁止按 slice 逐条构造；
// 多通道对合并后大段做 CalcDataSplitByPortGroup 比例切分。
class InsTempScatterRHD : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::RHD};
    InsTempScatterRHD() = default;
    explicit InsTempScatterRHD(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempScatterRHD() override;

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    std::string Describe() const override
    {
        std::string info = "Template of scatter RHD with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    void SetRoot(u32 root);

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const;
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const;
    // 非 2^N 多级流输入桥：交换 root（唯一持有数据方）的 ccl 由上游自然布局原地置换为 σ 布局
    // （跨层契约经 RhdUnitStartOfRank 换算；环循环节约一个元素暂存，排空舞步封口
    // 暂存的队列生命周期）。2^N（σ=恒等）零开销跳过。
    HcclResult RearrangeToSigma(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const;
    HcclResult RunRHD(
        const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
        const TemplateDataParams& tempAlgParams);
    // 单步单通道传输的上下文：CalcDataSplitByPortGroup 产出的三元组（成组产出、成组消费）
    // + repeat/channelIdx 循环上下文，收敛形参到 6 个以内
    struct RhdChannelSplit {
        const std::vector<u64>* elemCountOut = nullptr;
        const std::vector<u64>* sizeOut = nullptr;
        const std::vector<u64>* elemOffset = nullptr;
        u32 repeat = 0;
        u32 channelIdx = 0;
    };
    // 单步单通道传输：对合并后大段按端口组比例切分的连续子段构造单条 DataSlice
    HcclResult ScatterStep(
        const RhdScatterStep& step, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams, const RhdSegBytes& seg,
        const RhdChannelSplit& split) const;

    u64 processSize_{0};
    u64 count_{0};
    bool isDmaRead_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_SCATTER_RHD_H
