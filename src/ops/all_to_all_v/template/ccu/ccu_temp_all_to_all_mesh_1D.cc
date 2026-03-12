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
#include "kernel/ccu_kernel_all_to_all_mesh1d.h"
#include "ccu_temp_all_to_all_mesh_1D.h"

namespace ops_hccl {

CcuTempAlltoAllMesh1D::CcuTempAlltoAllMesh1D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

CcuTempAlltoAllMesh1D::~CcuTempAlltoAllMesh1D()
{
}

HcclResult CcuTempAlltoAllMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // kernel数量
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAlltoAllMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    typeSize_ = DataTypeSizeGet(param.all2AllDataDes.sendType);
    CcuKernelInfo kernelInfo;

    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAlltoAllMesh1D>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAlltoAllMesh1D>(subCommRanks_[0].size(),
                                                                        mySubCommRank_,
                                                                        param.isMc2, // loadFromMem_
                                                                        param,
                                                                        subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAlltoAllMeshMem2Mem1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

// device侧调用
void CcuTempAlltoAllMesh1D::InitInsAlgTemplate(
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

HcclResult CcuTempAlltoAllMesh1D::KernelRun(const OpParam& param,
                                            const TemplateDataParams& templateDataParams,
                                            const TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D] Run");
    buffInfo_ = templateDataParams.buffInfo;

    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    // 拿到input和output的首地址,和每片小数据的大小
    uint64_t curSendCounts = templateDataParams.count;
    uint64_t sliceSize = templateDataParams.sliceSize;

    uint32_t                                rankId    = myRank_;
    uint64_t                                repeatNumTmp  = templateDataParams.repeatNum;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t srcStride = sliceSize + sendStrideSize_;
    uint64_t dstStride = sliceSize + recvStrideSize_;

    uint64_t dataType_ = param.all2AllVDataDes.sendType;
    uint64_t dataTypeSize_ = SIZE_TABLE[dataType_];
    uint64_t sliceBias = templateDataParams.processedDataCount * dataTypeSize_;

    HCCL_DEBUG("[CcuTempAlltoAllMesh1D::KernelRun] Start");
    if (tempRankSize_ == 1) {
        // 遗留 不确定对不对
        DataSlice usrInSlice = DataSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, sliceSize);
        DataSlice usrOutSlice = DataSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, sliceSize);
        LocalCopy(templateResource.threads[0], usrInSlice, usrOutSlice);

        HCCL_DEBUG("[CcuTempAlltoAllMesh1D::KernelRun] end");
        return HcclResult::HCCL_SUCCESS;
    }

    uint64_t srcOffset = sliceBias;
    uint64_t dstOffset = sliceBias + myRank_ * dstStride;
    bool loadFromMem = false;

    HCCL_INFO("[CcuTempAllToAllMesh1D] Run Init: loadFromMem_[%d], myRank_[%d], dimSize[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        loadFromMem, myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, srcOffset, dstOffset);
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAlltoAllMesh1D>(
            inputAddr, outputAddr, sliceSize, token, srcOffset, dstOffset, srcStride, loadFromMem);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
    
    HCCL_DEBUG("[CcuTempAlltoAllMesh1D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAlltoAllMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}
} // namespace ops_hccl