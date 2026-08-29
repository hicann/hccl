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
#include "ccu_launch_dl.h"
#include "alg_template_base.h"
#include "ccu_kernel_reduce_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_nhr1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

std::vector<CostModelParam> CcuTempReduceScatterOmniPipeNHR1DMem2Mem::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = (param.portNum.size() == 1) ? param.portNum[0] : (param.portNum[0] + param.portNum[1]);
    int kernelNum = 1;
    int log2R = 0;
    for (u32 r = param.rankSize; r > 1; r >>= 1) {
        log2R++;
    }
    int taskNum = 4 * log2R + 1;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcNHRParams(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio, EngineType::CCU, B);
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::CCU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::CCU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

CcuTempReduceScatterOmniPipeNHR1DMem2Mem::CcuTempReduceScatterOmniPipeNHR1DMem2Mem(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    // 获取本卡在子通信域(如果有)中的rankid, 以及子通信域内所有卡数
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    templateRankSize_ = ranks.size();

    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] templateRankSize[%u]", __func__, rankId, mySubCommRank_, templateRankSize_);
}

CcuTempReduceScatterOmniPipeNHR1DMem2Mem::~CcuTempReduceScatterOmniPipeNHR1DMem2Mem() {}

HcclResult CcuTempReduceScatterOmniPipeNHR1DMem2Mem::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);

    // 需要1个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG(
        "[%s] notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__, resourceRequest.notifyNumOnMainThread,
        resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(strcpy_s(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuReduceScatterOmniPipeNHR1DMem2MemKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuReduceScatterOmniPipeNHR1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks_, channelDescs));
    std::vector<HcclChannelDesc> myChannelDescs;
    for (auto channel : channelDescs) {
        if (channel.channelProtocol == COMM_PROTOCOL_UB_CTP) {
            myChannelDescs.push_back(channel);
        }
    }
    HCCL_DEBUG("[%s] Get Clos Channel Success!", __func__);

    std::map<u32, u32> rank2ChannelIdx; // rankId和channel匹配
    for (u32 i = 0; i < myChannelDescs.size(); ++i) {
        u32 remoteRank = myChannelDescs[i].remoteRank;
        u32 subRankIdx = RemoteRankId2RankId(remoteRank);
        rank2ChannelIdx[subRankIdx] = i;
    }

    std::vector<NHRStepInfoRS> stepInfoVector;
    CHK_RET(CalcNHRInfo(stepInfoVector)); // NHR算法编排参数

    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->rank2ChannelIdx = rank2ChannelIdx;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);

    kernelInfo.channels = myChannelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] channelSize[%u] dimsize[%u] ccuKernelInfos.size[%u]", __func__, myRank_,
        mySubCommRank_, myChannelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterOmniPipeNHR1DMem2Mem::CalcNHRInfo(std::vector<NHRStepInfoRS>& stepInfoVector) const
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfoRS stepInfo;
        CHK_RET(GetStepInfo(step, stepInfo));
        stepInfoVector.push_back(stepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempReduceScatterOmniPipeNHR1DMem2Mem::GetNHRStepNum(u32 rankSize) const
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[%s] rankSize[%u] nSteps[%u]", __func__, rankSize, nSteps);
    return nSteps;
}

HcclResult CcuTempReduceScatterOmniPipeNHR1DMem2Mem::GetStepInfo(u32 step, NHRStepInfoRS& stepInfo) const
{
    // 将本rank号转换成算法使用的索引号
    u32 rankIdx = mySubCommRank_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 recvFrom = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 txSliceIdx = sendTo;
    u32 rxSliceIdx = rankIdx;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    // 计算本rank在本轮收/发中的slice编号
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx); // 虚拟id
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_DEBUG(
            "[%s] step[%u] myRank[%u] mySubCommRank[%u] recvFrom[%u] sendTo[%u] slice-i[%u] txSliceIdx[%u] "
            "rxSliceIdx[%u]",
            __func__, step, myRank_, mySubCommRank_, recvFrom, sendTo, i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuTempReduceScatterOmniPipeNHR1DMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempReduceScatterOmniPipeNHR1DMem2Mem::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    (void)param;
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = buffInfo_.inBuffBaseOff;
    uint64_t outBuffBaseOff = buffInfo_.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    if (localCopyFlag == 0) {
        uint64_t sliceStride = stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();
        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];
            std::vector<uint64_t> inputOmniSliceStrideVec;
            std::vector<uint64_t> inputOmniSliceSizeVec;
            for (uint32_t ridx = 0; ridx < stepSliceInfo.inputOmniPipeSliceStride.size(); ridx++) {
                inputOmniSliceStrideVec.push_back(stepSliceInfo.inputOmniPipeSliceStride[ridx][rpt]);
                inputOmniSliceSizeVec.push_back(stepSliceInfo.stepSliceSize[ridx][rpt]);
                HCCL_DEBUG(
                    "[%s] myRank[%u] stepSliceInfo.inputOmniPipeSliceStride[%d][%d]:%d", __func__, myRank_, ridx, rpt,
                    stepSliceInfo.inputOmniPipeSliceStride[ridx][rpt]);
                HCCL_DEBUG(
                    "[%s] myRank[%u] stepSliceInfo.stepSliceSize[%d][%d]:%d", __func__, myRank_, ridx, rpt,
                    stepSliceInfo.stepSliceSize[ridx][rpt]);
            }
            uint64_t inputSliceStride = templateDataParams.inputSliceStride;

            std::vector<uint64_t> taskArgs
                = {inputAddr, outputAddr, token, sliceSize, sliceStride, localCopyFlag, inputOmniPipeSliceStride};
            taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());
            taskArgs.push_back(inputSliceStride);
            taskArgs.insert(taskArgs.end(), inputOmniSliceSizeVec.begin(), inputOmniSliceSizeVec.end());
            HCCL_INFO(
                "[%s] myRank[%u] mySubCommRank[%u] rpt[%u] inputAddrBase[%llu] outputAddrBase[%llu] "
                "inBuffBaseOff[%llu] outBuffBaseOff[%llu] inputAddr[%llu] "
                "outputAddr[%llu] sliceSize[%llu] sliceStride[%llu] inputOmniPipeSliceStride[%llu] "
                "inputSliceStride[%llu]",
                __func__, myRank_, mySubCommRank_, rpt, inputAddrBase, outputAddrBase, inBuffBaseOff, outBuffBaseOff,
                inputAddr, outputAddr, sliceSize, sliceStride, inputOmniPipeSliceStride, inputSliceStride);
            CcuResult launchRet = HcommCcuKernelLaunch(
                templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), taskArgs.size());
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR(
                    "[CcuTempReduceScatterOmniPipeNHR1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d",
                    launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 语义改为返回当前template的类型，mesh返回1，nhr返回0
u64 CcuTempReduceScatterOmniPipeNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // 不需要Scratch buff
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempReduceScatterOmniPipeNHR1DMem2Mem::GetThreadNum() const { return 1; }

HcclResult CcuTempReduceScatterOmniPipeNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl
