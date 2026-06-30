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
#include "ccu_kernel_all_to_all_v_mesh1d_multi_jetty.h"
#include "ccu_temp_all_to_all_v_mesh_1D_multi_jetty.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {
constexpr uint32_t CONST_1 = 1;
constexpr uint32_t CONST_4 = 4;
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;

CcuTempAllToAllVMesh1DMultiJetty::CcuTempAllToAllVMesh1DMultiJetty(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    myRank_ = rankId;
    templateRankSize_ = subCommRanks[0].size();
}

CcuTempAllToAllVMesh1DMultiJetty::~CcuTempAllToAllVMesh1DMultiJetty()
{
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAllToAllVMesh1DMultiJetty");
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAllToAllVMesh1DMultiJettyKernel);

    std::vector<HcclChannelDesc> channelDescs;
    // 计算建链诉求以COMM_TOPO_1DMESH为优先级，优先建mesh链，没有mesh链建clos链
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));

    std::vector<uint32_t> jettyNums;
    CHK_RET(SetJettyNums(jettyNums, false));
    jettyNums_ = jettyNums;
    auto kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1DMultiJetty>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = myRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelArg->jettyNums = jettyNums;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::AddTaskArgA2AInfo(A2ASendRecvInfo &localSendRecvInfo, std::vector<uint64_t> &taskArgs)
{   
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
    std::vector<uint32_t> jettyNums;
    CHK_RET(SetJettyNums(jettyNums, false));
    jettyNums_ = jettyNums;
    uint64_t rankSize = jettyNums_.size();
    for (uint64_t i = 0; i < rankSize; i++) {
        uint64_t blockSize = UB_MAX_TRANS_SIZE;
        uint64_t loopNum = UINT64_MAX - (localSendRecvInfo.sendLength[i] / blockSize + 1) * jettyNums_[i];
        uint64_t sliceSize = 0;
        uint64_t tailSliceSize = 0;
        if (localSendRecvInfo.sendLength[i] >= blockSize) {
            sliceSize = blockSize / jettyNums_[i] / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
            tailSliceSize = blockSize - sliceSize * (jettyNums_[i] - 1);        
            if (jettyNums_[i] == 1) {
                sliceSize = 0;
                tailSliceSize = blockSize;
            }
        }
        uint64_t lastBlockSize = localSendRecvInfo.sendLength[i] % blockSize;
        uint64_t lastSliceSize = lastBlockSize / jettyNums_[i] / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        uint64_t lastTailSliceSize = lastBlockSize - lastSliceSize * (jettyNums_[i] - 1);
        if (jettyNums_[i] == 1) {
            lastSliceSize = 0;
            lastTailSliceSize = lastBlockSize;
        }

        uint64_t sendOffset = localSendRecvInfo.sendOffset[i];        
        uint64_t recvOffset = localSendRecvInfo.recvOffset[i];        
        taskArgs.push_back(sliceSize);
        taskArgs.push_back(tailSliceSize);
        taskArgs.push_back(lastSliceSize);
        taskArgs.push_back(lastTailSliceSize);
        taskArgs.push_back(loopNum);
        taskArgs.push_back(sendOffset);
        taskArgs.push_back(recvOffset);

        auto tailGoSize = CalGoSize(lastBlockSize, config);
        for (auto val : tailGoSize) {
            taskArgs.push_back(val);
        }
        HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty] rankIdx[%llu] " \
            "localSendRecvInfo.sendLength[%llu]," \
            "localSendRecvInfo.sendOffset[%llu]," \
            "localSendRecvInfo.recvLength[%llu]," \
            "localSendRecvInfo.recvOffset[%llu]", i,
            localSendRecvInfo.sendLength[i],
            localSendRecvInfo.sendOffset[i],
            localSendRecvInfo.recvLength[i],
            localSendRecvInfo.recvOffset[i]);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t srcOffset = 0; // alltoallv假设src起始地址为发送rank的对应块起始地址
    uint64_t dstOffset = 0; // alltoallv假设dst起始地址为接收rank的对应块起始地址

    HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty] inputAddr[%llu], outputAddr[%llu],"
              "srcOffset[%llu], dstOffset[%llu]",
              inputAddr, outputAddr, srcOffset, dstOffset);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, srcOffset, dstOffset};

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(UB_MAX_TRANS_SIZE, config);
    for (auto val : xnMaxTransportGoSize) {
        taskArgs.push_back(val);
    }
    CHK_RET(AddTaskArgA2AInfo(localSendRecvInfo_, taskArgs));
    uint64_t argSize = taskArgs.size();

    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
        taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllToAllVMesh1DMultiJetty::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, 
                            token, srcOffset, dstOffset, xnMaxTransportGoSize[0], xnMaxTransportGoSize[1], 
                            xnMaxTransportGoSize[2], xnMaxTransportGoSize[3], templateRankSize_));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

// executor在orchestra中调用
void CcuTempAllToAllVMesh1DMultiJetty::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::FastLaunch(const OpParam& param,
    const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    uint64_t argSize = 9;
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputOffsetIdx = 0;
    constexpr u32 outputOffsetIdx = 1;
    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    HCCL_INFO("[CcuTempAlltoAllVMesh1DMultiJetty::FastLaunch]: inputAddr[%llu], outputAddr[%llu], "
              "srcOffset[%llu], dstOffset[%llu]", args[0], args[1], args[3], args[4]);
    A2ASendRecvInfo localSendRecvInfo;
    HcclDataType dataType = param.all2AllVDataDes.sendType;
    uint64_t dataTypeSize = SIZE_TABLE[dataType];
    templateRankSize_ = static_cast<uint32_t>(args[9]);
    localSendRecvInfo.sendCounts.resize(templateRankSize_, 0);
    localSendRecvInfo.sendDispls.resize(templateRankSize_, 0);
    localSendRecvInfo.sendLength.resize(templateRankSize_, 0);
    localSendRecvInfo.sendOffset.resize(templateRankSize_, 0);
    localSendRecvInfo.recvCounts.resize(templateRankSize_, 0);
    localSendRecvInfo.recvDispls.resize(templateRankSize_, 0);
    localSendRecvInfo.recvLength.resize(templateRankSize_, 0);
    localSendRecvInfo.recvOffset.resize(templateRankSize_, 0);
    const u64* data = reinterpret_cast<const u64*>(param.varData);
    for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * templateRankSize_; i++) {
        u64 val = i / templateRankSize_;
        u64 curRank = i % templateRankSize_;
        switch(val) {
            case 0:
                localSendRecvInfo.sendLength[curRank] = data[i] * dataTypeSize;
                break;
            case 1:
                localSendRecvInfo.recvLength[curRank] = data[i] * dataTypeSize;
                break;
            case 2:
                localSendRecvInfo.sendOffset[curRank] = data[i] * dataTypeSize;
                break;
            case 3:
                localSendRecvInfo.recvOffset[curRank] = data[i] * dataTypeSize;
                break;
            default:
                break;
        }
    }
    std::vector<uint64_t> taskArgsVec(args, args + argSize);
    CHK_RET(AddTaskArgA2AInfo(localSendRecvInfo, taskArgsVec));
    argSize = taskArgsVec.size();
    void *taskArgs = reinterpret_cast<void*>(taskArgsVec.data());
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    HCCL_INFO("[CcuTempAllToAllVMesh1DMultiJetty::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllToAllVMesh1DMultiJetty::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::SetJettyNums(std::vector<uint32_t>& jettyNums, const bool multijetty) const
{
    jettyNums.resize(templateRankSize_, 0);
    for (int i = 0; i < templateRankSize_; i++) {
        if (i == myRank_) {
            jettyNums[i] = CONST_1;
        } else if (multijetty) {
            jettyNums[i] = CONST_4;
        } else {
            jettyNums[i] = CONST_1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl