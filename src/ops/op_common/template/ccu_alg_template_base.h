/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_ALG_TEMPLATE_BASE
#define HCCL_CCU_ALG_TEMPLATE_BASE

#include "common_alg_template_base.h"
#include <array>
#include <set>
#include <string>

namespace ops_hccl {

constexpr uint32_t CCU_DIE_NUM_MAX_2 = 2;
constexpr uint32_t MAX_KERNEL_NUM_2DIE = 3;
constexpr uint32_t KERNEL_FULLMESH = 0;
constexpr uint32_t KERNEL_CLOS_MAJOR = 1;
constexpr uint32_t KERNEL_CLOS_MINOR = 2;

class CcuAlgTemplateBase : public CommonAlgTemplateBase {
public:
    explicit CcuAlgTemplateBase();
    explicit CcuAlgTemplateBase(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);

    ~CcuAlgTemplateBase() override;

    std::string Describe() const override = 0;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;

    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    uint64_t PointerToAddr(void* pointer) const;

    HcclResult GetToken(const BuffInfo& buffinfo, uint64_t& token) const;

    static HcclResult
    GetChannelDieId(HcclComm comm, uint32_t rankId, const HcclChannelDesc& channelDesc, uint32_t& dieId);
    static HcclResult
    GetChannelBwCoeff(HcclComm comm, uint32_t rankId, const HcclChannelDesc& channelDesc, uint32_t& bwCoeff);
    static HcclResult RestoreChannelMap(
        const std::vector<HcclChannelDesc>& channelDescs,
        std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);

    static HcclResult SelectChannelToVec(
        const HcclComm comm, const u32 myRankId, const u32 rmtRankId,
        const std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc, const u32 dieId,
        std::map<u32, u32>& rank2ChannelIdx, std::vector<HcclChannelDesc>& channels);
    static HcclResult ReverseChannelPerDieIfNeed(
        const HcclComm comm, const u32 myRankId, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);
    static HcclResult GetDieInfoFromChannelDescs(
        HcclComm comm, const std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc, u32 myRankId,
        uint32_t& dieNum, uint32_t& dieId);
    static HcclResult CalcDieSplitRatio(
        HcclComm comm, uint32_t myRank, bool is2Plus6, const std::vector<HcclChannelDesc>& majorChs,
        const std::vector<HcclChannelDesc>& minorChs, double& ratio);
    static HcclResult SplitChannelsByDie(
        HcclComm comm, uint32_t myRank, std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc,
        std::map<uint32_t, std::vector<HcclChannelDesc>>& singleChByDie,
        std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie, bool& is2Plus6,
        std::set<u32>* closPeers = nullptr);
    static HcclResult PartitionChannelsFor2Die(
        const std::map<uint32_t, std::vector<HcclChannelDesc>>& singleChByDie,
        const std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie, bool is2Plus6, uint32_t myRank,
        uint32_t& kernelCount, uint32_t& fullmeshDieId,
        std::array<std::vector<HcclChannelDesc>, MAX_KERNEL_NUM_2DIE>& kernelChannels,
        std::array<std::vector<u32>, MAX_KERNEL_NUM_2DIE>& kernelRankGroup, const std::string& tag);

protected:
    OpMode opMode_ = OpMode::OPBASE;
    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 root_ = 0;
    u32 templateRankSize_ = 0;
    uint64_t scratchBufferSize_ = 0;
    HcclDataType dataType_ = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp_ = HcclReduceOp::HCCL_REDUCE_RESERVED;
    BuffInfo buffInfo_{};
    std::vector<std::vector<u32>> subCommRanks_;
};
} // namespace ops_hccl
#endif // HCCLV2_CCU_ALG_TEMPLATE_BASE
