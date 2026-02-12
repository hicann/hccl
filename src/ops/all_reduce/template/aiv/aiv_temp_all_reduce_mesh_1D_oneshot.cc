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
#include "aiv/aiv_temp_all_reduce_mesh_1D_oneshot.h"

namespace ops_hccl {

AivTempAllReduceMesh1DOneShot::AivTempAllReduceMesh1DOneShot(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                                       const std::vector<std::vector<u32>> &subCommRanks)
                                                       : AivAlgTemplateBase(param, rankId, subCommRanks)
{
}

AivTempAllReduceMesh1DOneShot::~AivTempAllReduceMesh1DOneShot()
{
}

u64 AivTempAllReduceMesh1DOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    u64 scratchMultiple = tempRankSize_;
    return scratchMultiple;
}

HcclResult AivTempAllReduceMesh1DOneShot::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
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

HcclResult AivTempAllReduceMesh1DOneShot::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void) dataSize;
    if (numBlocksLimit >= (tempRankSize_ + 1)) {
        numBlocks = tempRankSize_ + 1;
    } else {
        // 如果要用更少的核心可以在这里折算，比如rankSize/2个核心
        numBlocks = numBlocksLimit;
    }
    HCCL_INFO("[AivTempAllReduceMesh1DOneShot] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllReduceMesh1DOneShot::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& tempAlgParams,
                                                 const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAllReduceMesh1DOneShot] KernelRun start");

    IncSliceId();  // 自动增长sliceId，传入aivCountTag
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivAllReduceArgs;
    aivAllReduceArgs.cmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    aivAllReduceArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivAllReduceArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivAllReduceArgs.rank = u32(myRank_);
    aivAllReduceArgs.rankSize = tempRankSize_;
    aivAllReduceArgs.count = tempAlgParams.sliceSize / SIZE_TABLE[dataType_];
    aivAllReduceArgs.dataType = dataType_;
    aivAllReduceArgs.op = param.reduceType;
    aivAllReduceArgs.root = root_;
    aivAllReduceArgs.aivCountTag = (static_cast<uint32_t>(param.aivCountTag) << AIV_TAG_MOVE_LEFT_BITS) | static_cast<uint32_t>(sliceId_);  // 传入aivCountTag，Lauch时重新组装为aivCountTag
    aivAllReduceArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivAllReduceArgs.stream = param.stream;
    aivAllReduceArgs.isOpBase = (param.opMode == OpMode::OPBASE);
    aivAllReduceArgs.xRankSize = subCommRanks_[0].size();
    aivAllReduceArgs.yRankSize = 0;
    aivAllReduceArgs.zRankSize = 0;
    for (u32 i = 0; i < subCommRanks_[0].size(); i++){
        aivAllReduceArgs.topo_[i] = subCommRanks_[0][i];
    }
    if (subCommRanks_.size() > 1){
        aivAllReduceArgs.yRankSize = subCommRanks_[1].size();
        for (u32 i = 0; i < subCommRanks_[1].size(); i++){
            aivAllReduceArgs.topo_[TOPO_LEN_Y_OFFSET + i] = subCommRanks_[1][i];
        }
    }
    if (subCommRanks_.size() == MAX_DIM_NUM){
        aivAllReduceArgs.zRankSize = subCommRanks_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < subCommRanks_[MAX_DIM_NUM - 1].size(); i++){
            aivAllReduceArgs.topo_[TOPO_LEN_Z_OFFSET + i] = subCommRanks_[MAX_DIM_NUM - 1][i];
        }
    }

    CHK_RET(CalNumBlocks(aivAllReduceArgs.numBlocks, tempAlgParams.sliceSize, param.numBlocksLimit));

    aivAllReduceArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivAllReduceArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivAllReduceArgs.repeatNum = tempAlgParams.repeatNum;
    aivAllReduceArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAllReduceArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    ExecuteKernelLaunch(aivAllReduceArgs);

    HCCL_INFO("[AivTempAllReduceMesh1DOneShot] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl