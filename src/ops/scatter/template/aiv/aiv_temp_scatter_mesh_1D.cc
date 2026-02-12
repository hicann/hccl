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
#include "aiv/aiv_temp_scatter_mesh_1D.h"

namespace ops_hccl {

AivTempScatterMesh1D::AivTempScatterMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                                       const std::vector<std::vector<u32>> &subCommRanks)
                                                       : AivAlgTemplateBase(param, rankId, subCommRanks)
{
}

AivTempScatterMesh1D::~AivTempScatterMesh1D()
{
}

HcclResult AivTempScatterMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
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

HcclResult AivTempScatterMesh1D::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& tempAlgParams,
                                                 const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempScatterMesh1D] KernelRun start");

    IncSliceId();  // 自动增长sliceId，传入aivCountTag
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivScatterArgs;
    aivScatterArgs.cmdType = HcclCMDType::HCCL_CMD_SCATTER;
    aivScatterArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivScatterArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivScatterArgs.rank = u32(myRank_);
    aivScatterArgs.rankSize = tempRankSize_;
    aivScatterArgs.count = tempAlgParams.sliceSize / SIZE_TABLE[dataType_];
    aivScatterArgs.dataType = dataType_;
    aivScatterArgs.op = param.reduceType;
    aivScatterArgs.root = root_;
    aivScatterArgs.aivCountTag = (static_cast<uint32_t>(param.aivCountTag) << AIV_TAG_MOVE_LEFT_BITS) | static_cast<uint32_t>(sliceId_);  // 传入aivCountTag，Lauch时重新组装为aivCountTag  // todo
    aivScatterArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivScatterArgs.stream = param.stream;
    aivScatterArgs.isOpBase = (param.opMode == OpMode::OPBASE);
    aivScatterArgs.xRankSize = subCommRanks_[0].size();
    aivScatterArgs.yRankSize = 0;
    aivScatterArgs.zRankSize = 0;
    for (u32 i = 0; i < subCommRanks_[0].size(); i++){
        aivScatterArgs.topo_[i] = subCommRanks_[0][i];
    }
    if (subCommRanks_.size() > 1){
        aivScatterArgs.yRankSize = subCommRanks_[1].size();
        for (u32 i = 0; i < subCommRanks_[1].size(); i++){
            aivScatterArgs.topo_[TOPO_LEN_Y_OFFSET + i] = subCommRanks_[1][i];
        }
    }
    if (subCommRanks_.size() == MAX_DIM_NUM){
        aivScatterArgs.zRankSize = subCommRanks_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < subCommRanks_[MAX_DIM_NUM - 1].size(); i++){
            aivScatterArgs.topo_[TOPO_LEN_Z_OFFSET + i] = subCommRanks_[MAX_DIM_NUM - 1][i];
        }
    }

    aivScatterArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivScatterArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivScatterArgs.repeatNum = tempAlgParams.repeatNum;
    aivScatterArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivScatterArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    ExecuteKernelLaunch(aivScatterArgs);

    HCCL_INFO("[AivTempScatterMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl