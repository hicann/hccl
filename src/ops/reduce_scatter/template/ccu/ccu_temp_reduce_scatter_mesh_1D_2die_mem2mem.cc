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
#include "ccu_kernel_reduce_scatter_mesh1d_2die_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_2die_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

constexpr u32 DIE_NUM = 2;
constexpr u32 UDIE0 = 0;
constexpr u32 UDIE1 = 1;

CcuTempReduceScatterMeshMem2Mem1D2Die::CcuTempReduceScatterMeshMem2Mem1D2Die(const OpParam& param, const u32 rankId,
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

CcuTempReduceScatterMeshMem2Mem1D2Die::~CcuTempReduceScatterMeshMem2Mem1D2Die()
{
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes] "
                "notifyNumOnMainThread[%u] slaveThreadNum[%u] ccuKernelNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, resourceRequest.ccuKernelNum[0]);
    
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<std::vector<HcclChannelDesc>> channelDescsVec;
    std::vector<std::vector<u32>>             subRankGroup;
    bool isReduceToOutput = false;

    channelDescsVec.resize(DIE_NUM);
    subRankGroup.resize(DIE_NUM);

    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    for (auto channel : channelDescs) {
        uint32_t dieId = 0;
        CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempAllReduceMesh1DMem2Mem2DieOneShot][CalcRes] dieId is invalid"), HCCL_E_INTERNAL);
        channelDescsVec[dieId].push_back(channel);
        subRankGroup[dieId].push_back(channel.remoteRank);
    }

    subRankGroup[UDIE1].push_back(myRank_);

    uint32_t tmpDieId   = myRank_ < subRankGroup[UDIE0].back() ? 0 : 1;
    localReduceOffset_  = subRankGroup[1 - tmpDieId][0];

    for (int dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuKernelInfo kernelInfo;
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuReduceScatterMesh1D2DieMem2MemKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceScatterMesh1D2DieMem2MemKernel);

        isReduceToOutput = channelDescsVec[dieId].size() > channelDescsVec[1 - dieId].size() ? false : true;

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1D2DieMem2Mem>();
        kernelArg->gRankSize = templateRankSize_;
        kernelArg->rankSize = subRankGroup[dieId].size();
        kernelArg->isReduceToOutput = isReduceToOutput;
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subRankGroup = subRankGroup[dieId];
        kernelArg->subCommRanks = subCommRanks_;
        kernelInfo.setKernelArg(kernelArg);

        HCCL_INFO("[CcuTempReduceScatterMeshMem2Mem1D2Die]gRankSize[%d], rankSize[%d], myRank[%d], isReduceToOutput[%d]",
            templateRankSize_, subRankGroup[dieId].size(), myRank_, isReduceToOutput);
        kernelInfo.channels = channelDescsVec[dieId];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D2Die::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t repeatNumTmp       = templateDataParams.repeatNum;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize    = templateDataParams.sliceSize;
    uint64_t lastSliceSize      = templateDataParams.tailSize;

    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;

    // 原GeneArgs中计算: currentRankSliceInputOffset = inputSliceStride * rankId_ (rankId_=myRank_)
    uint64_t currentRankSliceInputOffset = inputSliceStride * myRank_;

    LoopGroupConfig config{};
    config.msInterleave = REDUCE_MS_CNT;
    config.loopCount = 16;
    config.memSlice = CCU_MS_SIZE;
    auto goSize = CalGoSize(normalSliceSize, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, scratchAddr,
                                      currentRankSliceInputOffset, inputRepeatStride, outputRepeatStride,
                                      normalSliceSize, repeatNum,
                                      goSize[0], goSize[1], goSize[2], goSize[3]};

    HCCL_INFO("[CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "scratchAddr[%llu], currentRankSliceInputOffset[%llu], inputRepeatStride[%llu], "
              "outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], repeatNum[%llu]",
              inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, inputRepeatStride,
              outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumTmp);

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for(uint64_t dieIdx = 0; dieIdx < DIE_NUM; dieIdx++) {
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[dieIdx], templateResource.ccuKernels[dieIdx],
                                                    taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempReduceScatterMeshMem2Mem1D2Die::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    // 使用SDMA inlinereduce做localcopy scratch -> output
    uint64_t hcclBuffOffset = buffInfo_.hcclBuffBaseOff;
    hcclBuffOffset += normalSliceSize * (templateRankSize_ / DIE_NUM);
    DataSlice srcSlice(buffInfo_.hcclBuff.addr, hcclBuffOffset, normalSliceSize, normalSliceSize / DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, normalSliceSize, normalSliceSize / DATATYPE_SIZE_TABLE[param.DataDes.dataType]);
    CHK_RET(LocalReduce(templateResource.threads[0], srcSlice, dstSlice, param.DataDes.dataType, param.reduceType));

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterMeshMem2Mem1D2Die::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 CcuTempReduceScatterMeshMem2Mem1D2Die::GetThreadNum() const
{
    return DIE_NUM;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D2Die::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl