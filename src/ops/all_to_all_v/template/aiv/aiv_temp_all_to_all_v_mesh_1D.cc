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
constexpr u32 MAX_NUM_BLOCKS_ALL_TO_ALL_V = 48; // 算法不交付控核

AivTempAlltoAllVMesh1D::AivTempAlltoAllVMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                                       const std::vector<std::vector<u32>> &subCommRanks)
                                                       : AivAlgTemplateBase(param, rankId, subCommRanks)
{
}

AivTempAlltoAllVMesh1D::~AivTempAlltoAllVMesh1D()
{
}

HcclResult AivTempAlltoAllVMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
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
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllVMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void) dataSize;
    numBlocks = numBlocksLimit;
    HCCL_INFO("[AivTempAlltoAllVMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllVMesh1D::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& tempAlgParams,
                                                 const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAlltoAllVMesh1D] KernelRun start");

    IncSliceId();  // 自动增长sliceId，传入aivCountTag
    dataType_ = param.all2AllVDataDes.sendType;
    AivOpArgs aivAlltoAllVArgs;
    aivAlltoAllVArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    aivAlltoAllVArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivAlltoAllVArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivAlltoAllVArgs.rank = u32(myRank_);
    aivAlltoAllVArgs.rankSize = tempRankSize_;
    aivAlltoAllVArgs.count = tempAlgParams.sliceSize / SIZE_TABLE[dataType_];
    aivAlltoAllVArgs.dataType = dataType_;
    aivAlltoAllVArgs.op = param.reduceType;
    aivAlltoAllVArgs.root = root_;
    aivAlltoAllVArgs.aivCountTag = (static_cast<uint32_t>(param.aivCountTag) << AIV_TAG_MOVE_LEFT_BITS) | static_cast<uint32_t>(sliceId_);  // 传入aivCountTag，Lauch时重新组装为aivCountTag  // todo
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

    u64 dataSize = tempAlgParams.inputSliceStride;
    CHK_RET(CalNumBlocks(aivAlltoAllVArgs.numBlocks, dataSize, MAX_NUM_BLOCKS_ALL_TO_ALL_V));

    aivAlltoAllVArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivAlltoAllVArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivAlltoAllVArgs.repeatNum = tempAlgParams.repeatNum;
    aivAlltoAllVArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAlltoAllVArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    ExecuteKernelLaunch(aivAlltoAllVArgs);

    HCCL_INFO("[AivTempAlltoAllVMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl