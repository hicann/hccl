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
#include "alg_data_trans_wrapper.h"
#include "ccu_kernel_all_reduce_mesh_1D_2die_oneshot.h"
#include "ccu_temp_all_reduce_mesh_1D_2die_oneshot.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {
constexpr u32 ALL_REDUCE_DIE_NUM = 2;
constexpr u32 DIE_WORK = 2;

CcuTempAllreduceMesh1D2DieOneShot::CcuTempAllreduceMesh1D2DieOneShot(const OpParam& param, const u32 rankId,
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

CcuTempAllreduceMesh1D2DieOneShot::~CcuTempAllreduceMesh1D2DieOneShot()
{
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
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
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        channelDescsDie[dieId].push_back(channel);
        groupRanksforDie[dieId].push_back(channel.remoteRank);
        HCCL_INFO("[CcuTempAllreduceMesh1D2DieOneShot::calRes] dieId[%d],channelIdx[%d]", dieId, channelIdx);
        channelIdx++;
    }

    for (uint32_t die = 0; die < ALL_REDUCE_DIE_NUM; die++) {
        CcuKernelInfo kernelInfo;
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAllreduceMesh1D2DieOneShot");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllreduceMesh1D2DieOneShotKernel);

        bool rmtReduceWithMyRank = channelDescsDie[die].size() > channelDescsDie[1 - die].size() ? false : true;
        if (rmtReduceWithMyRank) {
            groupRanksforDie[die].push_back(mySubCommRank_);
        }

        auto kernelArg = std::make_shared<CcuKernelArgAllreduceMesh1D2DieOneShot>();
        kernelArg->rankSize = groupRanksforDie[die].size();
        kernelArg->rankId = mySubCommRank_;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = groupRanksforDie;
        kernelArg->rmtReduceWithMyRank = rmtReduceWithMyRank;
        
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channelDescsDie[die];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
        
        HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] channelDescs.size()=%llu, rankSize=%llu, ",
            channelDescsDie[die].size(), groupRanksforDie[die].size());
    }

    HCCL_DEBUG("[CcuTempAllreduceMesh1D2DieOneShot::CalcRes] ccuKernelInfos.size()=%llu",
            resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllreduceMesh1D2DieOneShot::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t sliceSize = templateDataParams.sliceSize;

    uint32_t dieNum = ALL_REDUCE_DIE_NUM;

    uint64_t scratchBaseOffset0 = 0;
    uint64_t scratchBaseOffset1 = sliceSize;

    uint32_t dataTypeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t localRedcueSize0 = ((sliceSize / dataTypeSize) / DIE_WORK) * dataTypeSize;
    uint64_t localRedcueSize1 = sliceSize - localRedcueSize0;

    uint64_t localReduceSliceOffset0 = 0;
    uint64_t localReduceSliceOffset1 = localRedcueSize0;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE;

    auto rmtReduceGoSize = CalGoSize(sliceSize, config);
    auto localReduceGoSize0 = CalGoSize(localRedcueSize0, config);
    auto localReduceGoSize1 = CalGoSize(localRedcueSize1, config);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        scratchAddr,
        scratchBaseOffset0,
        scratchBaseOffset1,
        localReduceSliceOffset0,
        localReduceSliceOffset1
    };

    for (auto &goSize : {rmtReduceGoSize, localReduceGoSize0, localReduceGoSize1}) {
        for (auto &element : goSize) {
            taskArgs.push_back(element);
        }
    }
    uint64_t argSize = taskArgs.size();

    for (auto dieId = 0; dieId < dieNum; dieId++) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieId], templateResource.ccuKernels[dieId],
            taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllreduceMesh1D2DieOneShot::KernelRun] die[%d] kernel launch failed, ccuRet -> %d", dieId, launchRet);
            return ConvertCcuToHccl(launchRet);
        }
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
    (void)inBuffType;
    (void)outBuffType;
    return ALL_REDUCE_DIE_NUM;
}
} // namespace ops_hccl
