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
#include "template_utils.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh1d.h"
#include "ccu_temp_all_to_all_v_mesh_1D.h"
#include "ccu_launch_dl.h"

#define CONST_ZERO 0
#define CONST_ONE 1
#define CONST_TWO 2
#define CONST_THREE 3

namespace ops_hccl {

constexpr u32 DIE_0 = 0;
constexpr u32 DIE_1 = 1;
constexpr u32 DIE_NUM_1 = 1;
constexpr u32 DIE_NUM_2 = 2;

CcuTempAlltoAllVMesh1D::CcuTempAlltoAllVMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    loadFromMem_ = param.isMc2;
}

CcuTempAlltoAllVMesh1D::~CcuTempAlltoAllVMesh1D()
{
}

HcclResult CcuTempAlltoAllVMesh1D::CalcChannelRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, std::vector<HcclChannelDesc>& channelDescs)
{
    if (topoInfo->topoLevelNums > 1) {
        // 跨框场景全连接建链
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAlltoAllVMesh1D");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAlltoAllVMesh1DKernel);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRes(comm, param, topoInfo, channelDescs));
    
    auto kernelArg = std::make_shared<CcuKernelArgAlltoAllVMesh1D>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->loadFromMem = param.isMc2;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

// executor在orchestra中调用
void CcuTempAlltoAllVMesh1D::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

// device侧调用
void CcuTempAlltoAllVMesh1D::InitInsAlgTemplate(
    std::vector<u64> &sendCounts, std::vector<u64> &recvCounts,
    std::vector<u64> &sdispls, std::vector<u64> &rdispls)
{
    sendCounts_ = sendCounts;
    recvCounts_ = recvCounts;
    sdispls_ = sdispls;
    rdispls_ = rdispls;

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, sendCounts[%u] is [%u]", i, sendCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, recvCounts[%u] is [%u]", i, recvCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, sdispls[%u] is [%u]", i, sdispls[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, rdispls[%u] is [%u]", i, rdispls[i]);
    }

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, sendCounts_[%u] is [%u]", i, sendCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, recvCounts_[%u] is [%u]", i, recvCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, sdispls_[%u] is [%u]", i, sdispls_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D] InitInsAlgTemplate, rdispls_[%u] is [%u]", i, rdispls_[i]);
    }
}

HcclResult CcuTempAlltoAllVMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAlltoAllVMesh1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[CcuTempAlltoAllVMesh1D::FastLaunch] start");

    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    uint64_t rankSize = args[5];
    uint32_t myRank = static_cast<uint32_t>(args[6]);
    HcclDataType dataType = param.all2AllVDataDes.sendType;
    uint64_t dataTypeSize =  SIZE_TABLE[dataType];
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64),
    HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] param.varMemSize [%llu] is invalid", param.varMemSize), HCCL_E_PARA);
    
    A2ASendRecvInfo localSendRecvInfo;
    localSendRecvInfo.recvCounts.resize(rankSize, 0);
    localSendRecvInfo.recvDispls.resize(rankSize, 0);
    localSendRecvInfo.recvLength.resize(rankSize, 0);
    localSendRecvInfo.recvOffset.resize(rankSize, 0);
    localSendRecvInfo.sendCounts.resize(rankSize, 0);
    localSendRecvInfo.sendDispls.resize(rankSize, 0);
    localSendRecvInfo.sendLength.resize(rankSize, 0);
    localSendRecvInfo.sendOffset.resize(rankSize, 0);

    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize ; i++) {
        u64 val = i / rankSize;
 	    u64 curRank = i % rankSize;
        switch(val) {
            case CONST_ZERO:
                localSendRecvInfo.sendLength[curRank] = data[i] * dataTypeSize;
                break;
            case CONST_TWO:
                localSendRecvInfo.sendOffset[curRank] = data[i] * dataTypeSize;
                break;
            case CONST_THREE:
                localSendRecvInfo.recvOffset[curRank] = data[i] * dataTypeSize;
                break;
            default:
                break;
        }
    }

    uint64_t inputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[7];
    uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[8];
    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, args[2], args[3], args[4]};

    LoopGroupConfig  config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_INTERLEAVE;
    config.memSlice     = CCU_MS_INTERLEAVE * CCU_MS_SIZE;

    if (loadFromMem_) {
        taskArgs.push_back(0);  // 空地址占位，保证参数个数与load个数一致
    } else {
        uint64_t xnMaxTransportSize   = UB_MAX_TRANS_SIZE;
        auto     xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize, config);
        for (auto val : xnMaxTransportGoSize) {
            taskArgs.push_back(val);
        }

        for (uint64_t i = 0; i < rankSize; i++) {
            uint64_t tailSize = localSendRecvInfo.sendLength[i] % UB_MAX_TRANS_SIZE;
            uint64_t loopNum = UINT64_MAX - 1 - (localSendRecvInfo.sendLength[i] / UB_MAX_TRANS_SIZE);
            uint64_t sendOffset = localSendRecvInfo.sendOffset[i];
            uint64_t recvOffset = localSendRecvInfo.recvOffset[i];
            
            taskArgs.push_back(tailSize);
            taskArgs.push_back(loopNum);
            taskArgs.push_back(sendOffset);
            taskArgs.push_back(recvOffset);
            std::vector<uint64_t> virTailSize;
            virTailSize.resize(ALL_TO_ALL_V_VECTOR_NUM, 0);
            if (i == myRank) {
                auto tailGoSize = CalGoSize(tailSize, config);
                for (auto val : tailGoSize) {
                    taskArgs.push_back(val);
                }
            } else {
                for (auto val : virTailSize) {
                    taskArgs.push_back(val);
                }
            }
        }
    }
    
    uint64_t argSize = taskArgs.size();

    HCCL_INFO("[CcuTempAlltoAllVMesh1D::FastLaunch]: inputPtr[%llu], outputPtr[%llu],srcOffset[%llu],"
        " dstOffset[%llu], rankSize[%llu], myRank[%lu]", PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr),
        PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr), args[3], args[4], args[5], myRank);

    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                        tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                        taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    
    HCCL_INFO("[CcuTempAlltoAllVMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D::KernelRun(const OpParam& param, 
                                            const TemplateDataParams& templateDataParams,
                                            TemplateResource& templateResource)
{
    // 遗留：localSendRecvInfo_ 从哪里传入呢？
    HCCL_INFO("[CcuTempAlltoAllVMesh1D] KernelRun");

    buffInfo_ = templateDataParams.buffInfo;
    uint64_t totalSliceSize = localSendRecvInfo_.sendLength[0];

    if (tempRankSize_ == 1) {
        // ccu-alltoall算子的单P场景单独处理
        CHK_PRT_RET(localSendRecvInfo_.sendLength[myRank_] != localSendRecvInfo_.recvLength[myRank_],
                    HCCL_INFO("[CcuTempAlltoAllVMesh1D] rankSize = 1, sendLength[%llu] and recvLength[%llu]"
                               "should be equal.",
                               localSendRecvInfo_.sendLength[myRank_], localSendRecvInfo_.recvLength[myRank_]),
                    HcclResult::HCCL_E_PARA);
        CHK_PRT_RET(localSendRecvInfo_.sendLength[myRank_] == 0,
                    HCCL_INFO("[CcuTempAlltoAllVMesh1D] Single Rank and DataSlice size is 0, no need to process."),
                    HcclResult::HCCL_SUCCESS);

        DataSlice usrInSlice = DataSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, localSendRecvInfo_.sendLength[myRank_]);
        DataSlice usrOutSlice = DataSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, localSendRecvInfo_.sendLength[myRank_]);
        LocalCopy(templateResource.threads[0], usrInSlice, usrOutSlice);

        HCCL_INFO("[CcuTempAlltoAllVMesh1D] rankSize = 1, use InsLocalCopy for sliceSize[%llu].",
                  localSendRecvInfo_.sendLength[myRank_]);
        return HcclResult::HCCL_SUCCESS;
    }

    uint32_t                                rankId    = myRank_;

    std::vector<uint64_t> sliceSize;
    sliceSize.reserve(localSendRecvInfo_.sendLength.size());

    for (auto &slice : localSendRecvInfo_.sendLength) {
        sliceSize.push_back(slice);
    }

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t srcOffset = 0; // alltoallv假设src起始地址为发送rank的对应块起始地址
    uint64_t dstOffset = 0; // alltoallv假设dst起始地址为接收rank的对应块起始地址
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint32_t rankSize = tempRankSize_;

    HCCL_INFO("[CcuTempAllToAllVMesh1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        myRank_, tempRankSize_, inputAddr, outputAddr, sliceSize, srcOffset, dstOffset);
    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, srcOffset, dstOffset};

    LoopGroupConfig  config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_INTERLEAVE;
    config.memSlice     = CCU_MS_INTERLEAVE * CCU_MS_SIZE;

    if (loadFromMem_) {
        taskArgs.push_back(0);  // 空地址占位，保证参数个数与load个数一致
    } else {
        uint64_t xnMaxTransportSize   = UB_MAX_TRANS_SIZE;
        auto     xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize, config);
        for (auto val : xnMaxTransportGoSize) {
            taskArgs.push_back(val);
        }

        for (uint64_t i = 0; i < rankSize; i++) {
            uint64_t tailSize = localSendRecvInfo_.sendLength[i] % UB_MAX_TRANS_SIZE;
            uint64_t loopNum = UINT64_MAX - 1 - (localSendRecvInfo_.sendLength[i] / UB_MAX_TRANS_SIZE);
            uint64_t sendOffset = localSendRecvInfo_.sendOffset[i];
            uint64_t recvOffset = localSendRecvInfo_.recvOffset[i];
            
            taskArgs.push_back(tailSize);
            taskArgs.push_back(loopNum);
            taskArgs.push_back(sendOffset);
            taskArgs.push_back(recvOffset);
            std::vector<uint64_t> virTailSize;
            virTailSize.resize(ALL_TO_ALL_V_VECTOR_NUM, 0);
            if (i == myRank_) {
                auto tailGoSize = CalGoSize(tailSize, config);
                for (auto val : tailGoSize) {
                    taskArgs.push_back(val);
                }
            } else {
                for (auto val : virTailSize) {
                    taskArgs.push_back(val);
                }
            }
        }
    }

    uint64_t argSize = taskArgs.size();

    CcuResult launchRet =  HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    
    CcuKernelSubmitInfo subCommInfo;
    subCommInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(subCommInfo, inputAddr, outputAddr, 
        token, srcOffset, dstOffset, rankSize, myRank_, 
        buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
    templateResource.submitInfos.push_back(subCommInfo);    

    HCCL_DEBUG("[CcuTempReduceScatterMesh1D::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}
} // namespace ops_hccl