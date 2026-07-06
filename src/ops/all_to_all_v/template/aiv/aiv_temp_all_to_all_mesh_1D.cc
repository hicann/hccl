/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "hccl_aiv_utils.h"
#include "aiv/aiv_temp_all_to_all_mesh_1D.h"

namespace ops_hccl {

AivTempAlltoAllMesh1D::AivTempAlltoAllMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                                       const std::vector<std::vector<u32>> &subCommRanks)
                                                       : AivAlgTemplateBase(param, rankId, subCommRanks)
{
}

AivTempAlltoAllMesh1D::~AivTempAlltoAllMesh1D()
{
}

HcclResult AivTempAlltoAllMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                               AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    OpParam param_ = param;

    std::vector<HcclChannelDesc> level0Channels;
    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs, true));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UB_MEM) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[AivTempAlltoAllMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void) dataSize;
    HCCL_INFO("[AivTempAlltoAllMesh1D] Limit core num[%u]", numBlocksLimit);

    // 小于1的场景
    if (numBlocksLimit < 1) {
        numBlocks = numBlocksLimit;
        return HcclResult::HCCL_SUCCESS;
    }

    // rankSize在部分范围时，最多使用指定倍数个核
    constexpr u32 RANK_SIZE_CORE_CAP_THRESHOLD = 8;
    constexpr u32 MAX_CORE_MULTIPLE_OF_RANK_SIZE = 4;
    if (tempRankSize_ == RANK_SIZE_CORE_CAP_THRESHOLD) {
        u32 maxBlocks = MAX_CORE_MULTIPLE_OF_RANK_SIZE * tempRankSize_;
        numBlocksLimit = std::min(numBlocksLimit, maxBlocks);
    }

    if (numBlocksLimit >= tempRankSize_) {
        numBlocks = numBlocksLimit / tempRankSize_ * tempRankSize_;
    } else {
        u32 rankPerCore = (tempRankSize_ + numBlocksLimit - 1) / numBlocksLimit;  // 向上取整
        numBlocks = (tempRankSize_ + rankPerCore - 1) / rankPerCore;  // 向上取整
    }

    HCCL_INFO("[AivTempAlltoAllMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllMesh1D::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                                            const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAlltoAllMesh1D] KernelRun start");

    IncSliceId();  // 自动增长sliceId，传入sliceId
    dataType_ = param.all2AllVDataDes.sendType;
    AivOpArgs aivAlltoAllArgs;
    aivAlltoAllArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALL;
    aivAlltoAllArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivAlltoAllArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivAlltoAllArgs.rank = u32(myRank_);
    aivAlltoAllArgs.rankSize = tempRankSize_;

    aivAlltoAllArgs.count = tempAlgParams.sendCounts.front();
    HCCL_INFO("[AivTempAlltoAllMesh1D] KernelRun rank %d , input[%p] output[%p] count[%llu]",
              aivAlltoAllArgs.rank, aivAlltoAllArgs.input, aivAlltoAllArgs.output, aivAlltoAllArgs.count);
    aivAlltoAllArgs.dataType = dataType_;
    aivAlltoAllArgs.op = param.reduceType;
    aivAlltoAllArgs.root = root_;
    aivAlltoAllArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivAlltoAllArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivAlltoAllArgs.stream = param.stream;
    aivAlltoAllArgs.isOpBase = (param.opMode == OpMode::OPBASE);
    aivAlltoAllArgs.xRankSize = subCommRanks_[0].size();
    aivAlltoAllArgs.yRankSize = 0;
    aivAlltoAllArgs.zRankSize = 0;
    for (u32 i = 0; i < subCommRanks_[0].size(); i++){
        aivAlltoAllArgs.topo_[i] = subCommRanks_[0][i];
    }
    if (subCommRanks_.size() > 1){
        aivAlltoAllArgs.yRankSize = subCommRanks_[1].size();
        for (u32 i = 0; i < subCommRanks_[1].size(); i++){
            aivAlltoAllArgs.topo_[TOPO_LEN_Y_OFFSET + i] = subCommRanks_[1][i];
        }
    }
    if (subCommRanks_.size() == MAX_DIM_NUM){
        aivAlltoAllArgs.zRankSize = subCommRanks_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < subCommRanks_[MAX_DIM_NUM - 1].size(); i++){
            aivAlltoAllArgs.topo_[TOPO_LEN_Z_OFFSET + i] = subCommRanks_[MAX_DIM_NUM - 1][i];
        }
    }

    u64 dataSize = tempAlgParams.inputSliceStride;
    CHK_RET(CalNumBlocks(aivAlltoAllArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivAlltoAllArgs.inputSliceStride =
        reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts)[0] * DATATYPE_SIZE_TABLE[dataType_];
    aivAlltoAllArgs.outputSliceStride =
        reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts)[0] * DATATYPE_SIZE_TABLE[dataType_];
    aivAlltoAllArgs.repeatNum = tempAlgParams.repeatNum;
    aivAlltoAllArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAlltoAllArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivAlltoAllArgs));

    HCCL_INFO("[AivTempAlltoAllMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl