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
#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"
#include "ccu_kernel_all_reduce_mesh1d_mem2mem.h"

namespace ops_hccl {

CcuTempAllReduceMeshMem2Mem1D::CcuTempAllReduceMeshMem2Mem1D(const OpParam& param, 
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    templateRankSize_ = subCommRanks[0].size();
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType; 
}

CcuTempAllReduceMeshMem2Mem1D::~CcuTempAllReduceMeshMem2Mem1D()
{
}

uint64_t CcuTempAllReduceMeshMem2Mem1D::RoundUp(uint64_t dividend, uint64_t divisor) const
{
    return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
}

HcclResult CcuTempAllReduceMeshMem2Mem1D::CalcSlice(const u64 dataSize, RankSliceInfo &sliceInfoVec)
{
    std::vector<SliceInfo> tmp(subCommRanks_.size());
    sliceInfoVec.resize(templateRankSize_, tmp);

    u64 unitAllignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize      = RoundUp(dataSize, (templateRankSize_ * unitAllignSize)) * unitAllignSize;
    HCCL_INFO("chunkSize[%llu], dataSize[%llu], templateRankSize_[%u], unitAllignSize[%llu]", chunkSize, dataSize,
              templateRankSize_, unitAllignSize);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64       currChunkSize  = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice          = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR(
            "[CcuTempAllReduceMeshMem2Mem1D] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!",
            chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMeshMem2Mem1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                  AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllReduceMeshMem2Mem1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllReduceMeshMem2Mem1D>(arg);
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
        HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");
    }
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceMeshMem2Mem1D>(subCommRanks_[0].size(),
                                                                                    mySubCommRank_,
                                                                                    param,
                                                                                    subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMeshMem2Mem1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceMeshMem2Mem1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 CcuTempAllReduceMeshMem2Mem1D::GetThreadNum() const
{
    return 1;
}
 
HcclResult CcuTempAllReduceMeshMem2Mem1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMeshMem2Mem1D::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                    const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(templateDataParams.sliceSize, sliceInfoVec));

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize    = sliceInfoVec[0][0].size;
    uint64_t lastSliceSize      = sliceInfoVec[templateRankSize_ - 1][0].size;
    uint64_t mySliceSize        = sliceInfoVec[mySubCommRank_][0].size;
    uint64_t isInputOutputEqual = (inputAddr == outputAddr)? 1: 0;

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceMeshMem2Mem1D>(
        inputAddr, outputAddr, token, scratchAddr, inputSliceStride, outputSliceStride, inputRepeatStride,
            outputRepeatStride, normalSliceSize, lastSliceSize, mySliceSize, isInputOutputEqual);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl