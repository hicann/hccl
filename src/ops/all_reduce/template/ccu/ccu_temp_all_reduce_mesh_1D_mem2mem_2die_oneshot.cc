/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "ccu_kernel_all_reduce_mesh1d_mem2mem_2die_oneshot.h"
#include "ccu_temp_all_reduce_mesh_1D_mem2mem_2die_oneshot.h"
#include "ccu_kernel.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

constexpr u32 DIE_NUM = 2;
 
CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CcuTempAllReduceMesh1DMem2Mem2DieOneShot(const OpParam& param, const u32 rankId,
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
 
CcuTempAllReduceMesh1DMem2Mem2DieOneShot::~CcuTempAllReduceMesh1DMem2Mem2DieOneShot()
{
}
 
HcclResult CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
 
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
 
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
 
    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    std::vector<std::vector<HcclChannelDesc>> channelsForDie;
    std::vector<std::vector<uint32_t>> kernelRanks;
    channelsForDie.resize(DIE_NUM);
    kernelRanks.resize(DIE_NUM);
    for (auto channel : channelDescs) {
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot][CalcRes] dieId is invalid"), HCCL_E_INTERNAL);
        channelsForDie[dieId].push_back(channel);
        kernelRanks[dieId].push_back(channel.remoteRank);
    }
 
    for (uint32_t die = 0; die < DIE_NUM; die++) {
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                return std::make_unique<CcuKernelAllReduceMesh1DMem2Mem2DieOneShot>(arg);
                            };
        bool rmtReduceWithMyRank = channelsForDie[die].size() > channelsForDie[1 - die].size() ? false : true;
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    param,
                                                                                    kernelRanks[die],
                                                                                    subCommRanks_,
                                                                                    rmtReduceWithMyRank);
        kernelInfo.channels =  channelsForDie[die];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }
 
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult CcuTempAllReduceMesh1DMem2Mem2DieOneShot::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
 
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t normalSliceSize    = templateDataParams.sliceSize;
    
    // 双die模式，下发两个kernel
    for (uint64_t i = 0; i < DIE_NUM; i++) {
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceMesh1DMem2Mem2DieOneShot>(
            inputAddr, outputAddr, token, scratchAddr, normalSliceSize);
        void* taskArgPtr = static_cast<void*>(taskArg.get());
        HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[i], templateResource.ccuKernels[i], taskArgPtr);
    }
    
    return HcclResult::HCCL_SUCCESS;
}
 
u64 CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}
} // namespace ops_hccl