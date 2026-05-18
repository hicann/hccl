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
#include "ccu_kernel_scatter_mesh1d.h"
#include "ccu_temp_scatter_mesh1d.h"

namespace ops_hccl {

CcuTempScatterMesh1D::CcuTempScatterMesh1D(const OpParam &param, const u32 rankId,
                                                         const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    HCCL_INFO("Start to run CcuTempScatterMesh1D");
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
}

CcuTempScatterMesh1D::~CcuTempScatterMesh1D() {}

void CcuTempScatterMesh1D::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempScatterMesh1D][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::vector<u32> ranks = subCommRanks_[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + " "; }
    HCCL_INFO("[CcuTempScatterMesh1D][SetRoot] ranks = subCommRanks[0] is: %s", ranksStr.c_str());
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
}

HcclResult CcuTempScatterMesh1D::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                                AlgResourceRequest &resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempScatterMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;

    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelScatterMesh1D>(arg);
    };
    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempScatterMesh1D::CalcRes] Get Mesh Channel Success!");
    }
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgScatterMesh1D>(subCommRanks_[0].size(), mySubCommRank_,
                                                                              subCommRootId_, param, subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempScatterMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempScatterMesh1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempScatterMesh1D::FastLaunch] start");
    const uint64_t *args = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    CcuTaskArgScatterMesh1D taskArg(
            PointerToAddr(buffInfo_.inputPtr) + args[0],
            PointerToAddr(buffInfo_.outputPtr) + args[1],
            args[2],args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]);

    void* taskArgPtr = static_cast<void*>(&taskArg);

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0],
                                tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));

    HCCL_DEBUG("[CcuTempScatterMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
                                                  TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[CcuTempScatterMesh1D] sliceSize is 0, no need do, just success.");
        return HcclResult::HCCL_SUCCESS;
    }
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t repeatNumTmp = templateDataParams.repeatNum;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;

    uint64_t isInputOutputEqual = inputAddr == outputAddr ? 1 : 0;
    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;

    HCCL_INFO("[CcuTempScatterMesh1D] create CcuTaskArgScatterMesh1D, normalSliceSize [%u]", normalSliceSize);
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgScatterMesh1D>(
        inputAddr, outputAddr, token, inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride, normalSliceSize,
        lastSliceSize, repeatNum, isInputOutputEqual);

    void *taskArgPtr = static_cast<void *>(taskArg.get());

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    HCCL_DEBUG("[CcuTempScatterMesh1D::KernelRun] end");

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    submitInfo.cachedArgs[0]=buffInfo_.inBuffBaseOff; // input、output、scratch只存对应的偏移
    submitInfo.cachedArgs[1]=buffInfo_.outBuffBaseOff;
    submitInfo.cachedArgs[2]=token;
    submitInfo.cachedArgs[3]=inputSliceStride;
    submitInfo.cachedArgs[4]=outputSliceStride;
    submitInfo.cachedArgs[5]=inputRepeatStride;
    submitInfo.cachedArgs[6]=outputRepeatStride;
    submitInfo.cachedArgs[7]=normalSliceSize;
    submitInfo.cachedArgs[8]=lastSliceSize;
    submitInfo.cachedArgs[9]=repeatNum;
    submitInfo.cachedArgs[10]=isInputOutputEqual;
    templateResource.submitInfos.push_back(submitInfo);

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // scatter操作不需要scratch buffer
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

u64 CcuTempScatterMesh1D::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempScatterMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}
}  // namespace ops_hccl
