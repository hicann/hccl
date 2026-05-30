/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aiv_utils.h"
#include "aiv/aiv_temp_all_gather_mesh_1D.h"

namespace ops_hccl {

AivTempAllGatherMesh1D::AivTempAllGatherMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                                       const std::vector<std::vector<u32>> &subCommRanks)
                                                       : AivAlgTemplateBase(param, rankId, subCommRanks)
{
}

AivTempAllGatherMesh1D::~AivTempAllGatherMesh1D()
{
}

HcclResult AivTempAllGatherMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
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
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UB_MEM) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[AivTempAllGatherMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempAllGatherMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void) dataSize;
    numBlocks = numBlocksLimit;
    HCCL_INFO("[AivTempAllGatherMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllGatherMesh1D::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& tempAlgParams,
                                                 const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAllGatherMesh1D] KernelRun start");

    IncSliceId();  // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivAllGatherArgs;
    aivAllGatherArgs.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    aivAllGatherArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivAllGatherArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivAllGatherArgs.rank = u32(myRank_);
    aivAllGatherArgs.rankSize = tempRankSize_;
    aivAllGatherArgs.count = tempAlgParams.sliceSize / SIZE_TABLE[dataType_];
    aivAllGatherArgs.dataType = dataType_;
    aivAllGatherArgs.op = param.reduceType;
    aivAllGatherArgs.root = root_;
    aivAllGatherArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivAllGatherArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivAllGatherArgs.stream = param.stream;
    aivAllGatherArgs.isOpBase = (param.opMode == OpMode::OPBASE);
    aivAllGatherArgs.xRankSize = subCommRanks_[0].size();
    aivAllGatherArgs.yRankSize = 0;
    aivAllGatherArgs.zRankSize = 0;
    for (u32 i = 0; i < subCommRanks_[0].size(); i++){
        HCCL_INFO("[AivTempAllGatherMesh1D] subCommRanks_ 1, [%u]", subCommRanks_[0].size());
        aivAllGatherArgs.topo_[i] = subCommRanks_[0][i];
    }
    if (subCommRanks_.size() > 1){
        HCCL_INFO("[AivTempAllGatherMesh1D] subCommRanks_ 2");
        aivAllGatherArgs.yRankSize = subCommRanks_[1].size();
        for (u32 i = 0; i < subCommRanks_[1].size(); i++){
            aivAllGatherArgs.topo_[TOPO_LEN_Y_OFFSET + i] = subCommRanks_[1][i];
        }
    }
    if (subCommRanks_.size() == MAX_DIM_NUM){
        HCCL_INFO("[AivTempAllGatherMesh1D] subCommRanks_ 3");
        aivAllGatherArgs.zRankSize = subCommRanks_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < subCommRanks_[MAX_DIM_NUM - 1].size(); i++){
            aivAllGatherArgs.topo_[TOPO_LEN_Z_OFFSET + i] = subCommRanks_[MAX_DIM_NUM - 1][i];
        }
    }

    u64 dataSize = tempAlgParams.inputSliceStride;
    CHK_RET(CalNumBlocks(aivAllGatherArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivAllGatherArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivAllGatherArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivAllGatherArgs.repeatNum = tempAlgParams.repeatNum;
    aivAllGatherArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAllGatherArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivAllGatherArgs));

    HCCL_INFO("[AivTempAllGatherMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl