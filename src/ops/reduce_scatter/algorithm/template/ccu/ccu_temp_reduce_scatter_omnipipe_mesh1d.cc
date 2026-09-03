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
#include "ccu_kernel_reduce_scatter_omnipipe_mesh1d.h"
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

std::vector<CostModelParam> CcuTempReduceScatterOmniPipeMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = (param.portNum.size() == 1) ? param.portNum[0] : (param.portNum[0] + param.portNum[1]);
    int kernelNum = 1;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio, EngineType::CCU, B);
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::CCU, C);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

CcuTempReduceScatterOmniPipeMesh1D::CcuTempReduceScatterOmniPipeMesh1D(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    HCCL_DEBUG("[%s] templateRankSize:%u", __func__, templateRankSize_);
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u]", __func__, rankId, mySubCommRank_);
}

CcuTempReduceScatterOmniPipeMesh1D::~CcuTempReduceScatterOmniPipeMesh1D() {}

u64 CcuTempReduceScatterOmniPipeMesh1D::GetThreadNum() const { return 1; }

HcclResult CcuTempReduceScatterOmniPipeMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

uint32_t CcuTempReduceScatterOmniPipeMesh1D::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempReduceScatterOmniPipeMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;

    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG(
        "[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__, resourceRequest.notifyNumOnMainThread,
        resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;

    CHK_SAFETY_FUNC_RET(
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuReduceScatterOmniPipeMesh1DKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuReduceScatterOmniPipeMesh1DKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UB_CTP) {
                HCCL_ERROR(
                    "[CcuTempReduceScatterOmniPipeMesh1D][CalcRes] channelProtocol: %u", channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempReduceScatterOmniPipeMesh1D::CalcRes] Get Mesh Channel Success!");

    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterOmniPipeMesh1D>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);

    kernelInfo.channels = channelDescs;

    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] channelSize[%u] dimsize[%u] ccuKernelInfos.size[%u]", __func__, myRank_,
        mySubCommRank_, channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterOmniPipeMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    (void)param;
    buffInfo_ = templateDataParams.buffInfo;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = templateDataParams.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = templateDataParams.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    uint64_t offset = 0;
    if (localCopyFlag == 0) {
        uint64_t inputSliceStride = stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint64_t outputSliceStride = stepSliceInfo.stepOutputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();

        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];

            LoopGroupConfig config{};
            config.msInterleave = CCU_MS_INTERLEAVE;
            config.loopCount = CCU_MS_DEFAULT_LOOP_COUNT;
            config.memSlice = CCU_MS_SIZE;
            auto goSize = CalGoSize(sliceSize, config, GetCcuVersion());
            HCCL_INFO(
                "[%s] myRank[%u] mySubCommRank[%u] rpt[%u] inputAddrBase[%llu] outputAddrBase[%llu] "
                "inBuffBaseOff[%llu] outBuffBaseOff[%llu] inputAddr[%llu] outputAddr[%llu] "
                "sliceSize[%llu] inputSliceStride[%llu] localCopyFlag[%llu]",
                __func__, myRank_, mySubCommRank_, rpt, inputAddrBase, outputAddrBase, inBuffBaseOff, outBuffBaseOff,
                inputAddr, outputAddr, sliceSize, inputSliceStride, localCopyFlag);
            std::vector<uint64_t> taskArgs
                = {inputAddr, outputAddr,    scratchAddr,      sliceSize,         offset,
                   token,     localCopyFlag, inputSliceStride, outputSliceStride, inputOmniPipeSliceStride,
                   goSize[0], goSize[1],     goSize[2],        goSize[3]};

            CcuResult launchRet = HcommCcuKernelLaunch(
                templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), taskArgs.size());
            CHK_PRT_RET(
                launchRet != CCU_SUCCESS,
                HCCL_ERROR(
                    "[CcuTempReduceScatterOmniPipeMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet),
                ConvertCcuToHccl(launchRet));
        }
    } else if (localCopyFlag == 1) {
        DataSlice srcSlice(
            buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(
            buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_DEBUG(
            "[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
            "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterOmniPipeMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}
} // namespace ops_hccl
