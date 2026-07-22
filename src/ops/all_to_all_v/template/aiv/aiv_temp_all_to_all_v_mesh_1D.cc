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
#include "aiv/aiv_temp_all_to_all_v_mesh_1D.h"

namespace ops_hccl {

AivTempAlltoAllVMesh1D::AivTempAlltoAllVMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                                       const std::vector<std::vector<u32>> &subCommRanks)
                                                       : AivAlgTemplateBase(param, rankId, subCommRanks)
{
}

AivTempAlltoAllVMesh1D::~AivTempAlltoAllVMesh1D()
{
}

HcclResult AivTempAlltoAllVMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
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
        HCCL_DEBUG("[AivTempAlltoAllVMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllVMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void) dataSize;
    HCCL_INFO("[AivTempAlltoAllVMesh1D] Limit core num[%u]", numBlocksLimit);
    if (numBlocksLimit == 0) {
        HCCL_ERROR("[AivTempAlltoAllVMesh1D] numBlocksLimit is 0");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (numBlocksLimit < tempRankSize_) {
        numBlocks = numBlocksLimit;
    } else {
        // 多核一 rank，对齐到 rankSize 倍数
        numBlocks = numBlocksLimit / tempRankSize_ * tempRankSize_;
    }
    HCCL_INFO("[AivTempAlltoAllVMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllVMesh1D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAlltoAllVMesh1D] KernelRun start");

    IncSliceId();  // 自动增长sliceId，传入sliceId
    dataType_ = param.all2AllVDataDes.sendType;
    AivOpArgs aivAlltoAllVArgs;
    aivAlltoAllVArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    aivAlltoAllVArgs.input = reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivAlltoAllVArgs.output = reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivAlltoAllVArgs.rank = u32(myRank_);
    aivAlltoAllVArgs.rankSize = tempRankSize_;
    aivAlltoAllVArgs.count = tempAlgParams.sliceSize / SIZE_TABLE[dataType_];
    aivAlltoAllVArgs.dataType = dataType_;
    aivAlltoAllVArgs.op = param.reduceType;
    aivAlltoAllVArgs.root = root_;
    aivAlltoAllVArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivAlltoAllVArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivAlltoAllVArgs.stream = param.stream;
    aivAlltoAllVArgs.isOpBase = (param.opMode == OpMode::OPBASE);
    aivAlltoAllVArgs.xRankSize = subCommRanks_[0].size();
    aivAlltoAllVArgs.yRankSize = 0;
    aivAlltoAllVArgs.zRankSize = 0;

    size_t count = tempRankSize_;
    std::copy(tempAlgParams.sendCounts.data(),
              tempAlgParams.sendCounts.data() + count,
              aivAlltoAllVArgs.extraArgs.sendCounts);
    std::copy(tempAlgParams.sdispls.data(),
              tempAlgParams.sdispls.data() + count,
              aivAlltoAllVArgs.extraArgs.sendDispls);
    std::copy(tempAlgParams.recvCounts.data(),
              tempAlgParams.recvCounts.data() + count,
              aivAlltoAllVArgs.extraArgs.recvCounts);
    std::copy(tempAlgParams.rdispls.data(),
              tempAlgParams.rdispls.data() + count,
              aivAlltoAllVArgs.extraArgs.recvDispls);

    for (u32 i = 0; i < subCommRanks_[0].size(); i++){
        aivAlltoAllVArgs.topo_[i] = subCommRanks_[0][i];
    }
    if (subCommRanks_.size() > 1){
        aivAlltoAllVArgs.yRankSize = subCommRanks_[1].size();
        for (u32 i = 0; i < subCommRanks_[1].size(); i++){
            aivAlltoAllVArgs.topo_[TOPO_LEN_Y_OFFSET + i] = subCommRanks_[1][i];
        }
    }
    if (subCommRanks_.size() == MAX_DIM_NUM){
        aivAlltoAllVArgs.zRankSize = subCommRanks_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < subCommRanks_[MAX_DIM_NUM - 1].size(); i++){
            aivAlltoAllVArgs.topo_[TOPO_LEN_Z_OFFSET + i] = subCommRanks_[MAX_DIM_NUM - 1][i];
        }
    }

    u64 dataSize = tempAlgParams.sliceSize;
    CHK_RET(CalNumBlocks(aivAlltoAllVArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivAlltoAllVArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivAlltoAllVArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivAlltoAllVArgs.repeatNum = tempAlgParams.repeatNum;
    aivAlltoAllVArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAlltoAllVArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivAlltoAllVArgs));

    HCCL_INFO("[AivTempAlltoAllVMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl