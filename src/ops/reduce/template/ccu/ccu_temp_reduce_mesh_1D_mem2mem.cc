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
#include "ccu_kernel_reduce_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_mesh_1D_mem2mem.h"

namespace ops_hccl {

CcuTempReduceMesh1DMem2Mem::CcuTempReduceMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    auto rootIt = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), param.root);
    if (rootIt != subCommRanks[0].end()) {
        mySubCommRoot_ = std::distance(subCommRanks[0].begin(), rootIt);
    }
    dataType_ = param.DataDes.dataType;
}

CcuTempReduceMesh1DMem2Mem::~CcuTempReduceMesh1DMem2Mem()
{
}

void CcuTempReduceMesh1DMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        mySubCommRoot_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO("[CcuTempBroadcastMesh1DMem2Mem][SetRoot] myRank_ [%u], set root_ [%u] subCommRanks[%u]", mySubCommRank_, root, mySubCommRoot_);
}

HcclResult CcuTempReduceMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelReduceMesh1DMem2Mem>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceMesh1DMem2Mem>(subCommRanks_[0].size(),
                                                                             mySubCommRank_,
                                                                             mySubCommRoot_,
                                                                             param,
                                                                             subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& templateDataParams,
                                                 const TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceMesh1DMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }

    buffInfo_ = templateDataParams.buffInfo;

    uint64_t                               inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t                               outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    HCCL_INFO("buffInfo_.inputSize: %d", buffInfo_.inputSize);
    uint64_t                               token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                                                     static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t                               repeatNum          = templateDataParams.repeatNum;                                                                                 
    uint64_t                               inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t                               outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t                               normalSliceSize    = templateDataParams.sliceSize;
    uint64_t                               lastSliceSize      = templateDataParams.tailSize;
    uint64_t                               repeatNumVar       = UINT64_MAX - repeatNum;

    // 数据切分为sliceNum块，当数据量不能均匀切分时，后面smallDataSliceNum个数据块比前面bigDataSliceNum个数据块每块少1个数据
    uint64_t sliceNum   = templateRankSize_ - 1; // 最终切分的chunk数量
    uint64_t sliceCount = normalSliceSize / DataTypeSizeGet(dataType_);
    // bigDataSliceSize、smallDataSliceSize 为 chunkSize
    uint64_t bigDataSliceNum    = sliceCount % sliceNum;
    uint64_t bigDataSliceSize   = (sliceCount / sliceNum + 1) * DataTypeSizeGet(dataType_);
    uint64_t smallDataSliceNum  = sliceNum - sliceCount % sliceNum;
    uint64_t smallDataSliceSize = sliceCount / sliceNum * DataTypeSizeGet(dataType_);

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceMeshMem2Mem1D>(
        inputAddr, outputAddr, token, bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
    
    HCCL_DEBUG("[CcuTempReduceMesh1DMem2Mem::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempReduceMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

} // namespace ops_hccl 