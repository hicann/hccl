/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_alg_template_base.h"
#include "log.h"
#include "ccu_res_dl.h"
#include <iterator>
#include <utility>

namespace ops_hccl {
CcuAlgTemplateBase::CcuAlgTemplateBase()
{
}

CcuAlgTemplateBase::CcuAlgTemplateBase(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                       const std::vector<std::vector<u32>> &subCommRanks)
    : opMode_(param.opMode), myRank_(rankId), root_(param.root), subCommRanks_(subCommRanks)
{
    for (const auto &ranks : subCommRanks) {
        templateRankSize_ += ranks.size();
    }
}

CcuAlgTemplateBase::~CcuAlgTemplateBase()
{
}

HcclResult CcuAlgTemplateBase::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                       AlgResourceRequest& resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of CcuAlgTemplateBase::CalcRes!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuAlgTemplateBase::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                         TemplateResource& templateResource)
{
    (void)param;
    (void)templateDataParams;
    (void)templateResource;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of CcuAlgTemplateBase::KernelRun!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuAlgTemplateBase::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    (void)tempFastLaunchCtx;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of CcuAlgTemplateBase::FastLaunch!");
    return HcclResult::HCCL_E_INTERNAL;
}

u64 CcuAlgTemplateBase::GetThreadNum() const
{
    return 0;
}

HcclResult CcuAlgTemplateBase::GetRes(AlgResourceRequest& resourceRequest) const
{
    (void)resourceRequest;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

u64 CcuAlgTemplateBase::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

uint64_t CcuAlgTemplateBase::PointerToAddr(void* pointer) const
{
    if (pointer != nullptr) {
        return reinterpret_cast<uint64_t>(pointer);
    } else {
        return 0;
    }
}

HcclResult CcuAlgTemplateBase::RestoreChannelMap(const std::vector<HcclChannelDesc>& channelDescs,
                                                 std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    for (auto &channel: channelDescs) {
        u32 remoteRank = channel.remoteRank;
        rankIdToChannelDesc[remoteRank].push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::GetChannelDieId(HcclComm comm, uint32_t rankId, const HcclChannelDesc& channelDesc,
                                               uint32_t& dieId)
{
    EndpointAttrDieId tmpDieId{};
    uint32_t infoLen = sizeof(EndpointAttrDieId);
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rankId, &(channelDesc.localEndpoint), ENDPOINT_ATTR_DIE_ID, infoLen,
                                         &tmpDieId));
    dieId = tmpDieId;
    HCCL_INFO("[CcuAlgTemplateBase::GetChannelDieId] rank[%d]: get channel die id [%d]", rankId, dieId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::GetChannelBwCoeff(HcclComm comm, uint32_t rankId, const HcclChannelDesc& channelDesc,
                                               uint32_t& bwCoeff)
{
    EndpointAttrBwCoeff tmpBwCoeff{};
    uint32_t infoLen = sizeof(EndpointAttrBwCoeff);
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rankId, &(channelDesc.localEndpoint), ENDPOINT_ATTR_BW_COEFF, infoLen,
                                         &tmpBwCoeff));
    bwCoeff = tmpBwCoeff;
    HCCL_INFO("[CcuAlgTemplateBase::GetChannelBwCoeff] rank[%d]: get channel bwCoeff [%d]", rankId, bwCoeff);
    return HcclResult::HCCL_SUCCESS;
}

/* nhr算法，需要遍历得到的channelDesc，判断使用几个die，如果是1个die，则还需要得到dieId。
   以便于算法挑选相应dieId的channelDesc */
HcclResult CcuAlgTemplateBase::GetDieInfoFromChannelDescs(HcclComm comm, 
    const std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc, 
    u32 myRankId, uint32_t &dieNum, uint32_t &dieId)
{
    constexpr u32 LINK_NUM_1 = 1;
    constexpr u32 LINK_NUM_2 = 2;
    // 遍历每个对端有几个channel
    for (const auto& pair: rankIdToChannelDesc) {
        u32 rmtRankId = pair.first;
        const std::vector<HcclChannelDesc> &channels = pair.second;
        if (channels.size() == LINK_NUM_1) {
            dieNum = 1;
            GetChannelDieId(comm, myRankId, channels[0], dieId);
            HCCL_INFO("[CcuAlgTemplateBase::GetDieNumFromChannelDescs] only 1 channel, dieNum = 1, dieId = %u.", dieId);
            return HcclResult::HCCL_SUCCESS;
        } 
        
        if (channels.size() == LINK_NUM_2) {
            // 检查2个channel是否在2个die上
            uint32_t dieId0 = 0;
            uint32_t dieId1 = 0;
            GetChannelDieId(comm, myRankId, channels[0], dieId0);
            GetChannelDieId(comm, myRankId, channels[1], dieId1);
            if (dieId0 == dieId1) {
                dieNum = LINK_NUM_1;
                dieId = dieId0;
                HCCL_INFO("[CcuAlgTemplateBase::GetDieNumFromChannelDescs] 2 channels on the same die, dieNum = 1, dieId = %u.", dieId);
                return HcclResult::HCCL_SUCCESS;
            }
        } else {
            HCCL_ERROR("[CcuAlgTemplateBase::GetDieNumFromChannelDescs] get channelDescs fail: there are [%u] link to rank [%u]",
                    channels.size(), rmtRankId);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    dieNum = LINK_NUM_2;
    HCCL_INFO("[CcuAlgTemplateBase::GetDieNumFromChannelDescs] 2 channels on 2 dies, dieNum = 2.");
    return HcclResult::HCCL_SUCCESS;
}

/* 从rankIdToChannelDesc的map中，挑选出指定die上的channel加入到vec中，并将Index放入rank2ChannelIdx*/
HcclResult CcuAlgTemplateBase::SelectChannelToVec(const HcclComm comm, const u32 myRankId, const u32 rmtRankId,
    const std::map<u32, std::vector<HcclChannelDesc>> &rankIdToChannelDesc, const u32 dieId, 
    std::map<u32, u32>& rank2ChannelIdx, std::vector<HcclChannelDesc>& channels)
{
    auto it = rank2ChannelIdx.find(rmtRankId);
    if (it != rank2ChannelIdx.end() && channels.size() > it->second) {
        // 已经有对应channel，直接返回成功
        return HcclResult::HCCL_SUCCESS;
    }

    auto vecIt = rankIdToChannelDesc.find(rmtRankId);
    if (vecIt == rankIdToChannelDesc.end()) {
        // 不存在到对端rank的channel，报错
        HCCL_ERROR("[CcuAlgTemplateBase::SelectChannelToVec] there's no channel from rank[%u] to rank[%u]", myRankId, rmtRankId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    u32 curChannelIdx = channels.size();
    bool isFound = false;
    for (const HcclChannelDesc &channel: rankIdToChannelDesc.at(rmtRankId)) {
        uint32_t tmpDieId = 0;
        CHK_RET(GetChannelDieId(comm, myRankId, channel, tmpDieId));
        if (tmpDieId == dieId) {
            channels.push_back(channel);
            rank2ChannelIdx[rmtRankId] = curChannelIdx;
            isFound = true;
            break;
        }
    }
    if (!isFound) {
        HCCL_INFO("[CcuAlgTemplateBase::SelectChannelToVec] there's no channel from rank[%u] to rank[%u] on die[%u]",
             myRankId, rmtRankId, dieId);
    }
    HCCL_INFO("[CcuAlgTemplateBase::SelectChannelToVec] find channel rank[%u] to rank[%u] on die[%u] succeed."
        "Index=[%u]", myRankId, rmtRankId, dieId, curChannelIdx);
    return HcclResult::HCCL_SUCCESS;
}

/* 当2个die出框链路端口数不一致时，使用端口数（而非dieId）区分channel */
HcclResult CcuAlgTemplateBase::ReverseChannelPerDieIfNeed(const HcclComm comm, const u32 myRankId, 
    std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    if (channelsPerDie.size() <= 1) {
        HCCL_ERROR("[ReverseChannelPerDieIfNeed] channelsPerDie.size() = [%u], there's no channel on both dies", 
            channelsPerDie.size());
        return HCCL_E_PTR;
    }
    if (channelsPerDie[0].size() < 1 || channelsPerDie[1].size() < 1) {
        HCCL_ERROR("[ReverseChannelPerDieIfNeed] there's no channel in channelsPerDie");
        return HCCL_E_PTR;
    }
    uint32_t portNum0 = 0;
    uint32_t portNum1 = 0;
    GetChannelBwCoeff(comm, myRankId, channelsPerDie[0][0], portNum0);
    GetChannelBwCoeff(comm, myRankId, channelsPerDie[1][0], portNum1);

    if (portNum0 < portNum1) {
        // 2个die出框端口数不同，将端口数多的channel放在前面
        std::swap(channelsPerDie[0], channelsPerDie[1]);
    }
    HCCL_INFO("portNum0 = %lld,portNum1 = %lld",portNum0,portNum1);
    return HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::GetToken(const BuffInfo &buffinfo, uint64_t &token) const
{
    if (buffinfo.inputPtr != nullptr && buffinfo.inputSize != 0) {
        HcommCcuGetMemToken(PointerToAddr(buffinfo.inputPtr),
                            static_cast<uint64_t>(buffinfo.inputSize),
                            &token);
        return HCCL_SUCCESS;
    } else if (buffinfo.outputPtr != nullptr && buffinfo.outputSize != 0) {
        HcommCcuGetMemToken(PointerToAddr(buffinfo.outputPtr),
                            static_cast<uint64_t>(buffinfo.outputSize),
                            &token);
        return HCCL_SUCCESS;
    } else if (buffinfo.hcclBuff.addr != nullptr && buffinfo.hcclBuff.size != 0) {
        HcommCcuGetMemToken(PointerToAddr(buffinfo.hcclBuff.addr),
                            static_cast<uint64_t>(buffinfo.hcclBuff.size),
                            &token);
        return HCCL_SUCCESS;
    }
    HCCL_WARNING("[GetToken] inputMem, outputMem and hcclBuff are all null");
    return HCCL_E_PTR;
}

HcclResult CcuAlgTemplateBase::CalcDieSplitRatio(HcclComm comm, uint32_t myRank, bool is2Plus6,
    const std::vector<HcclChannelDesc>& majorChs,
    const std::vector<HcclChannelDesc>& minorChs, double& ratio)
{
    ratio = 1.0;
    if (is2Plus6 && !majorChs.empty() && !minorChs.empty()) {
        uint32_t majorBw = 0, minorBw = 0;
        CHK_RET(GetChannelBwCoeff(comm, myRank, majorChs[0], majorBw));
        CHK_RET(GetChannelBwCoeff(comm, myRank, minorChs[0], minorBw));
        if (majorBw + minorBw > 0) {
            ratio = static_cast<double>(majorBw) / (majorBw + minorBw);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::SplitChannelsByDie(HcclComm comm, uint32_t myRank,
    std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc,
    std::map<uint32_t, std::vector<HcclChannelDesc>>& singleChByDie,
    std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie,
    bool& is2Plus6, std::set<u32>* closPeers)
{
    using DieIdType = uint32_t;
    const uint32_t dieIdTypeSize = sizeof(DieIdType);
    for (auto& rankToChannels : rankIdToChannelDesc) {
        u32 remoteRank = rankToChannels.first;
        std::vector<HcclChannelDesc>& channelList = rankToChannels.second;
        bool isMulti = channelList.size() > 1;
        if (isMulti) {
            is2Plus6 = true;
            if (closPeers != nullptr) {
                closPeers->insert(remoteRank);
            }
        }
        for (const auto& channel : channelList) {
            DieIdType dieId = 0;
            EndpointDesc localEndpoint = channel.localEndpoint;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &localEndpoint, ENDPOINT_ATTR_DIE_ID,
                dieIdTypeSize, static_cast<void*>(&dieId)));
            (isMulti ? multiChByDie : singleChByDie)[dieId].emplace_back(channel);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::PartitionChannelsFor2Die(
    const std::map<uint32_t, std::vector<HcclChannelDesc>>& singleChByDie,
    const std::map<uint32_t, std::vector<HcclChannelDesc>>& multiChByDie,
    bool is2Plus6, uint32_t myRank, uint32_t& kernelCount, uint32_t& fullmeshDieId,
    std::array<std::vector<HcclChannelDesc>, MAX_KERNEL_NUM_2DIE>& kernelChannels,
    std::array<std::vector<u32>, MAX_KERNEL_NUM_2DIE>& kernelRankGroup,
    const std::string& tag)
{
    auto fillKernel = [&kernelChannels, &kernelRankGroup](uint32_t kernelIdx,
        const std::vector<HcclChannelDesc>& channels) {
        for (const auto& ch : channels) {
            kernelChannels[kernelIdx].emplace_back(ch);
            kernelRankGroup[kernelIdx].push_back(ch.remoteRank);
        }
    };

    if (is2Plus6) {
        kernelCount = MAX_KERNEL_NUM_2DIE;
        if (!singleChByDie.empty()) {
            fullmeshDieId = singleChByDie.begin()->first;
            fillKernel(KERNEL_FULLMESH, singleChByDie.at(fullmeshDieId));
        }
        kernelRankGroup[KERNEL_FULLMESH].push_back(myRank);
        for (auto& pair : multiChByDie) {
            fillKernel(pair.first == fullmeshDieId ? KERNEL_CLOS_MINOR : KERNEL_CLOS_MAJOR, pair.second);
        }
    } else {
        if (singleChByDie.size() < 2) {
            HCCL_ERROR("[%s][PartitionChannels] singleChByDie size[%zu] is less than 2, "
                "cannot partition channels for non-2Plus6 topology.", tag.c_str(), singleChByDie.size());
            return HcclResult::HCCL_E_INTERNAL;
        }
        kernelCount = MAX_KERNEL_NUM_2DIE - 1;
        auto it0 = singleChByDie.begin();
        auto it1 = std::next(it0);
        if (it0->second.size() > it1->second.size()) {
            std::swap(it0, it1);
        }
        fillKernel(KERNEL_FULLMESH, it0->second);
        kernelRankGroup[KERNEL_FULLMESH].push_back(myRank);
        fillKernel(KERNEL_CLOS_MAJOR, it1->second);
    }

    HCCL_INFO("[%s][PartitionChannels] Rank[%d], is2Plus6[%d], kernelCount[%u], "
        "fullmeshRankGroup[%zu], closMajorRankGroup[%zu], closMinorRankGroup[%zu].",
        tag.c_str(), myRank, is2Plus6, kernelCount, kernelRankGroup[KERNEL_FULLMESH].size(),
        kernelRankGroup[KERNEL_CLOS_MAJOR].size(), kernelRankGroup[KERNEL_CLOS_MINOR].size());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl