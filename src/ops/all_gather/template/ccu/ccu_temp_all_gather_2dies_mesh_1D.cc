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
#include "ccu_kernel_all_gather_2dies_mesh_1D.h"
#include "ccu_temp_all_gather_2dies_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
namespace ops_hccl {
constexpr u32 ALL_GATHER_DIE_NUM = 2;
CcuTempAllGather2DiesMesh1D::CcuTempAllGather2DiesMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}
 
CcuTempAllGather2DiesMesh1D::~CcuTempAllGather2DiesMesh1D()
{
}

HcclResult CcuTempAllGather2DiesMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{   
    //双die，声明一个从流
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(ALL_GATHER_DIE_NUM);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    uint32_t rankId = mySubCommRank_;
    EndpointAttrDieId tmpDieId {};

    HCCL_DEBUG("[CcuTempAllGather2DiesMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    std::vector<uint32_t> rankIdGroup0, rankIdGroup1;
    bool if0HandleSelfRank = true;
    CcuKernelInfo kernelInfo0, kernelInfo1;
    kernelInfo0.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelAllGather2DiesMesh1D>(arg);
    };
    kernelInfo1.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelAllGather2DiesMesh1D>(arg);
    };
    for (u32 j = 0; j < channelDescs.size(); j++) {
        CHK_RET(GetChannelDieId(comm, rankId, channelDescs[j], tmpDieId));
        if (tmpDieId == 0) {
            kernelInfo0.channels.push_back(channelDescs[j]);
            rankIdGroup0.push_back(channelDescs[j].remoteRank);
        } else {
            kernelInfo1.channels.push_back(channelDescs[j]);
            rankIdGroup1.push_back(channelDescs[j].remoteRank);
        }
    }
 
    if ((rankIdGroup0.size() > rankIdGroup1.size() && rankIdGroup1.size() != 0) || rankIdGroup0.size() == 0) {
        if0HandleSelfRank = false;
    }
    if (rankIdGroup0.size() != 0) {
        kernelInfo0.kernelArg = std::make_shared<CcuKernelArgAllGather2DiesMesh1D>(subCommRanks_[0].size(),
                                                                                         rankId, rankIdGroup0,
                                                                                         if0HandleSelfRank,
                                                                                         subCommRanks_,
                                                                                         param);
    }
    if (rankIdGroup1.size() != 0) {
        kernelInfo1.kernelArg = std::make_shared<CcuKernelArgAllGather2DiesMesh1D>(subCommRanks_[0].size(),
                                                                                         rankId, rankIdGroup1,
                                                                                         !if0HandleSelfRank,
                                                                                         subCommRanks_,
                                                                                         param);
    }
    if (rankIdGroup0.size() != 0) {
        resourceRequest.ccuKernelInfos.push_back(kernelInfo0);
    }
    if (rankIdGroup1.size() != 0) {
        resourceRequest.ccuKernelInfos.push_back(kernelInfo1);  
    }
    HCCL_DEBUG("[CcuTempAllGather2DiesMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult CcuTempAllGather2DiesMesh1D::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint32_t rankId = mySubCommRank_;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t sliceSize = templateDataParams.sliceSize;
    uint64_t offSet = rankId * templateDataParams.outputSliceStride;

    //前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    
    //双die模式，下发两个kernel
    for (uint64_t i = 0; i < ALL_GATHER_DIE_NUM; i++) {
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllGather2DiesMesh1D>(inputAddr, outputAddr,
                                                                                                         sliceSize, offSet, token);
        void* taskArgPtr = static_cast<void*>(taskArg.get());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[i], templateResource.ccuKernels[i], taskArgPtr));
        HCCL_DEBUG("[CcuTempAllGather2DiesMesh1D::KernelRun] end");
    }

    //后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGather2DiesMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

u64 CcuTempAllGather2DiesMesh1D::GetThreadNum() const
{
    return ALL_GATHER_DIE_NUM;
}
HcclResult CcuTempAllGather2DiesMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

}// namespace ops_hccl