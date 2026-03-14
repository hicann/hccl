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
#include "alg_template_base.h"
#include "kernel/ccu_kernel_all_to_all_mesh1d_multi_jetty.h"
#include "ccu_temp_all_to_all_mesh1d_multi_jetty.h"

namespace ops_hccl {
constexpr uint32_t STUB_JETTY_NUM = 4;
CcuTempAllToAllMesh1dMultiJetty::CcuTempAllToAllMesh1dMultiJetty(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    for (u32 i = 0; i < subCommRanks_.size(); i++) {
        for (u32 j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_INFO("subCommRanks_[%u][%u]=%u", i, j, subCommRanks_[i][j]);
        }
    }
    // 获取本卡在子通信域(如果有)中的rankid
    jettyNums_.assign(templateRankSize_, STUB_JETTY_NUM);
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllToAllMesh1dMultiJetty::~CcuTempAllToAllMesh1dMultiJetty()
{
}

HcclResult CcuTempAllToAllMesh1dMultiJetty::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllToAllMesh1DMultiJetty>(arg);
                         };

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllToAllMesh1DMultiJetty>(templateRankSize_,
                                                                                    myRank_,
                                                                                    param,
                                                                                    subCommRanks_,
                                                                                    jettyNums_);
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs,
                                                     CommTopo::COMM_TOPO_1DMESH));
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), templateRankSize_, resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1dMultiJetty::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;
    sendCounts_ = templateDataParams.sendCounts;
    recvCounts_ = templateDataParams.recvCounts;
    sdispls_ = templateDataParams.sdispls;
    rdispls_ = templateDataParams.rdispls;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(templateRankSize_);
    dataType_ = param.all2AllVDataDes.sendType;
    uint32_t dataTypeSize = SIZE_TABLE[dataType_];

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.inputPtr), buffInfo_.inputSize);
    uint64_t sliceSize    = templateDataParams.sliceSize;
    uint64_t totalSliceSize = (sdispls_[1] - sdispls_[0]) * dataTypeSize; // Bytes
    uint64_t srcStride = totalSliceSize;
    uint64_t dstStride = totalSliceSize;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = myRank_ * dstStride;
    HCCL_INFO("sliceSize=%llu, totalSliceSize=%llu, srcStride=%llu, dstStride=%llu, srcOffset=%llu, dstOffset=%llu,"
              " dataType_=%lu, dataTypeSize=%lu",
            sliceSize, totalSliceSize, srcStride, dstStride, srcOffset, dstOffset, dataType_,
            dataTypeSize);

    // 根据channel的jetty数量，再做切分
    std::vector<uint64_t> jettySlice, jettySliceTail;
    for (uint32_t rank = 0; rank < templateRankSize_; rank++) {
        // 128B对齐
        uint64_t quotient = sliceSize / jettyNums_[rank] / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        uint64_t tailSlice = sliceSize - quotient * (jettyNums_[rank] - 1);
        jettySlice.push_back(quotient);
        jettySliceTail.push_back(tailSlice);
    }

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllMesh1DMultiJetty>(
        inputAddr, outputAddr, sliceSize, jettySlice, jettySliceTail, token, srcOffset, dstOffset, srcStride);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
    
    HCCL_DEBUG("[CcuTempAllToAllMesh1dMultiJetty::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllToAllMesh1dMultiJetty::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 0;
}

} // namespace ops_hccl