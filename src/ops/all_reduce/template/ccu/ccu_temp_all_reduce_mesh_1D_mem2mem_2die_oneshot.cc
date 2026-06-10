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
#include "ccu_kernel_all_reduce_mesh1d_mem2mem_2die_oneshot.h"
#include "ccu_temp_all_reduce_mesh_1D_mem2mem_2die_oneshot.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

constexpr u32 DIE_NUM = 2;
 
CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CcuTempAllReduceMesh1DMem2Mem2DieOneShot(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
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
 
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
 
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
 
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
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAllReduceMesh1DMem2Mem2DieOneShot");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllReduceMesh1DMem2Mem2DieOneShotKernel);

        bool rmtReduceWithMyRank = channelsForDie[die].size() > channelsForDie[1 - die].size() ? false : true;

        auto kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot>();
        kernelArg->rankSize = subCommRanks_[0].size();
        kernelArg->rankId = mySubCommRank_;
        kernelArg->opParam = param;
        kernelArg->kernelRanks = kernelRanks[die];
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->rmtReduceWithMyRank = rmtReduceWithMyRank;
        
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channelsForDie[die];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }
 
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult CcuTempAllReduceMesh1DMem2Mem2DieOneShot::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t normalSliceSize = templateDataParams.sliceSize;

    uint32_t dataTypeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t localRedcueSize0 = (normalSliceSize / dataTypeSize) / DIE_NUM * dataTypeSize;
    uint64_t localRedcueSize1 = normalSliceSize - localRedcueSize0;

    uint64_t localReduceSliceOffset0 = 0;
    uint64_t localReduceSliceOffset1 = localRedcueSize0;

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        scratchAddr,
        normalSliceSize,
        localReduceSliceOffset0,
        localReduceSliceOffset1
    };

    LoopGroupConfig config{};
    constexpr uint32_t LOOP_NUM = 16;
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = LOOP_NUM;
    config.memSlice     = CCU_MS_SIZE;

    auto localReduceGoSize = CalGoSize(normalSliceSize, config);
    auto localReduceGoSize0 = CalGoSize(localRedcueSize0, config);
    auto localReduceGoSize1 = CalGoSize(localRedcueSize1, config);

    for (auto &element : localReduceGoSize) {
        taskArgs.push_back(element);
    }
    for (auto &element : localReduceGoSize0) {
        taskArgs.push_back(element);
    }
    for (auto &element : localReduceGoSize1) {
        taskArgs.push_back(element);
    }
    uint64_t argSize = taskArgs.size();
    
    for (uint64_t i = 0; i < DIE_NUM; i++) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[i], templateResource.ccuKernels[i],
            taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }
    
    return HcclResult::HCCL_SUCCESS;
}
 
u64 CcuTempAllReduceMesh1DMem2Mem2DieOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}
} // namespace ops_hccl
