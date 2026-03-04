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
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh1d.h"
#include "ccu_temp_all_to_all_v_mesh_1D.h"

namespace ops_hccl {

CcuTempAlltoAllVMesh1D::CcuTempAlltoAllVMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

CcuTempAlltoAllVMesh1D::~CcuTempAlltoAllVMesh1D()
{
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
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAlltoAllVMesh1D>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAlltoAllVMesh1D>(subCommRanks_[0].size(),
                                                                        mySubCommRank_,
                                                                        param.isMc2, // loadFromMem_
                                                                        param,
                                                                        subCommRanks_);
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
        HCCL_INFO("InitInsAlgTemplate, sendCounts[%u] is [%u]", i, sendCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, recvCounts[%u] is [%u]", i, recvCounts[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, sdispls[%u] is [%u]", i, sdispls[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, rdispls[%u] is [%u]", i, rdispls[i]);
    }

    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, sendCounts_[%u] is [%u]", i, sendCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, recvCounts_[%u] is [%u]", i, recvCounts_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, sdispls_[%u] is [%u]", i, sdispls_[i]);
    }
    for (u32 i = 0; i < templateRankSize_; i++) {
        HCCL_INFO("InitInsAlgTemplate, rdispls_[%u] is [%u]", i, rdispls_[i]);
    }
}

HcclResult CcuTempAlltoAllVMesh1D::KernelRun(const OpParam& param, 
                                            const TemplateDataParams& templateDataParams,
                                            const TemplateResource& templateResource)
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
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));

    HCCL_INFO("[CcuTempAllToAllVMesh1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        myRank_, tempRankSize_, inputAddr, outputAddr, sliceSize, srcOffset, dstOffset);
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAlltoAllVMesh1D>(
                inputAddr, outputAddr, sliceSize, token, srcOffset, 
                dstOffset, localSendRecvInfo_);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
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