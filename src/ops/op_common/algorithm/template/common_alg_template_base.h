/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_ALG_TEMPLATE_BASE_H
#define COMMON_ALG_TEMPLATE_BASE_H

#include "template_utils.h"
#include "alg_param.h"
#include "alg_parse.h"

namespace ops_hccl {

// Template 属性结构体，子类按需覆盖成员值（用于 cost model 等场景判断 template 特征）
struct TemplateProp {
    AlgoType algoType = AlgoType::UNKNOWN;
};

class CommonAlgTemplateBase {
public:
    explicit CommonAlgTemplateBase() = default;
    explicit CommonAlgTemplateBase(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    virtual ~CommonAlgTemplateBase() = default;

    // Template 属性，子类按需覆盖
    static constexpr TemplateProp props = {};

    // 模板 scratch 是否使用 acc+recv 双区、须与上游层级原地规约区隔离（DPU inter 只搬数：对端数据
    // 须落独立 recv 区、不能直接覆盖 acc 自身贡献，故 scratch 须排在上游输出区之后而非叠在其上）。
    // 默认 false（device 模板 write-reduce 同号槽、可在上游输出区内原地归约）；RhdDpuInter 特化为 true。
    // 作用域说明：本标志仅三级 executor（executor_3level.cc 基址/预算分支）消费。同一 RhdDpuInter
    // 模板也被 Dpu 二级注册（DpuReduceScatterSequenceMeshConcurRHD）绑定，但二级不消费本标志——
    // 二级 L1 的 hcclBuffBaseOff 恒 0（executor.cc 二级路径）、scratch 与上游交换同窗就地消费：
    // 二级只有一层上游（L0 终态即该层输出，无三层 executor 的 [0, R·C) 连乘原地规约区可叠），
    // acc/recv 与上游交换区按轮次全量覆盖、无跨层叠写冲突，且经 ST host_dpu_reducescatter_rhd 系列
    // 数据校验实证——故二级无需隔离、维持 hcclBuffBaseOff=0 口径。
    static constexpr bool kNeedsAccRecvScratchIsolation = false;

    virtual std::string Describe() const = 0;
    virtual HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest)
        = 0;
    virtual HcclResult GetRes(AlgResourceRequest& resourceRequest) const = 0;
    virtual u64 GetThreadNum() const = 0;
    virtual u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) = 0;

    virtual HcclResult
    KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
        = 0;
    virtual HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) = 0;
    virtual HcclResult CalcDataSplitByPortGroup(
        const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
        std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset)
    {
        CalcDataSplitByPortGroupCommon(
            totalDataCount, dataTypeSize, channels, elemCountOut, sizeOut, elemOffset, channelsPerRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    virtual HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
    {
        CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
        channelsPerRank_ = CalcChannelsPerRank(channels);
        return HCCL_SUCCESS;
    }

    // 强制设置每rank通道数，绕过按拓扑计算的结果（用于DETERMINISTIC_STRICT等确定性场景）
    void ForceSetChannelsPerRank(u32 channelsPerRank) { channelsPerRank_ = channelsPerRank; }

protected:
    u32 channelsPerRank_ = 1;
};

} // namespace ops_hccl

#endif // COMMON_ALG_TEMPLATE_BASE_H
