/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_kernel_all_reduce_mesh_1D_2die_oneshot.h"
#include "ccu_temp_all_reduce_mesh_1D_2die_oneshot.h"

namespace ops_hccl {
constexpr u32 ALL_REDUCE_DIE_NUM = 2;

CcuTempAllreduceMesh1D2DieOneShot::CcuTempAllreduceMesh1D2DieOneShot(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllreduceMesh1D2DieOneShot::~CcuTempAllreduceMesh1D2DieOneShot()
{
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 需要从流
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(ALL_REDUCE_DIE_NUM);
    HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<HcclChannelDesc> myChannelDescs;
    std::vector<std::vector<HcclChannelDesc>> channelDescsDie(resourceRequest.ccuKernelNum[0]);
    std::vector<std::vector<u32>> groupRanksforDie(resourceRequest.ccuKernelNum[0]);
    
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    uint32_t channelIdx = 0;
    for (auto channel : channelDescs) {
        if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
            myChannelDescs.push_back(channel);
        }
    }
    for (auto channel : myChannelDescs) {
        // 获取dieid
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        channelDescsDie[dieId].push_back(channel);
        groupRanksforDie[dieId].push_back(channel.remoteRank);
        HCCL_INFO("[CcuTempAllreduceMesh1D2DieOneShot::calRes] dieId[%d],channelIdx[%d]", dieId, channelIdx);
        channelIdx++;
    }

    for (uint32_t die = 0; die < ALL_REDUCE_DIE_NUM; die++) {
        // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                return std::make_unique<CcuKernelAllreduceMesh1D2DieOneShot>(arg);
                            };

        bool rmtReduceWithMyRank = channelDescsDie[die].size() > channelDescsDie[1 - die].size() ? false : true;
        if (rmtReduceWithMyRank) {
            groupRanksforDie[die].push_back(mySubCommRank_);
        }

        kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllreduceMesh1D2DieOneShot>(groupRanksforDie[die].size(),
                                                                                        mySubCommRank_,
                                                                                        param,
                                                                                        groupRanksforDie,
                                                                                        rmtReduceWithMyRank);
        kernelInfo.channels = channelDescsDie[die];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
            , channelDescsDie[die].size(), groupRanksforDie[die].size());
    }

    HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] ccuKernelInfos.size()=%llu",
            resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    std::vector<uint64_t> dimSize;
    dimSize.push_back(templateRankSize_);

    uint32_t                                rankId    = myRank_;
    uint64_t                                repeatNumTmp  = templateDataParams.repeatNum;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t sliceSize    = templateDataParams.sliceSize;

    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;
    uint32_t dieNum             = ALL_REDUCE_DIE_NUM;

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllreduceMesh1D2DieOneShot>(
        inputAddr, outputAddr, token, scratchAddr, sliceSize);

    void* taskArgPtr = static_cast<void*>(taskArg.get());
    // 同步
    for (auto dieId = 0; dieId < dieNum; dieId++) {
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[dieId], templateResource.ccuKernels[dieId], taskArgPtr));
        HCCL_INFO("[CcuTempAllreduceMesh1D2DieOneShot::KernelRun] die[%d] end", dieId);
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllreduceMesh1D2DieOneShot::GetThreadNum() const
{
    return ALL_REDUCE_DIE_NUM;
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;

    return HCCL_SUCCESS;
}

u64 CcuTempAllreduceMesh1D2DieOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return ALL_REDUCE_DIE_NUM;
}
} // namespace ops_hccl