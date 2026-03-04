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
#include "ccu_kernel_reduce_mesh2d_mem2mem.h"
#include "ccu_temp_reduce_mesh_2D_mem2mem.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

constexpr uint32_t AXIS_NUM_2 = 2;

CcuTempReduceMeshMem2Mem2D::CcuTempReduceMeshMem2Mem2D(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    dimSize_.emplace_back(subCommRanks[0].size());
    dimSize_.emplace_back(subCommRanks[1].size());
    myRankId_ = rankId;
    rootId_   = param.root;
    dataType_ = param.DataDes.dataType;
}

CcuTempReduceMeshMem2Mem2D::~CcuTempReduceMeshMem2Mem2D()
{
}

HcclResult CcuTempReduceMeshMem2Mem2D::PartitionChannels(const std::vector<HcclChannelDesc>& channelDescs,
                                                        std::vector<HcclChannelDesc>& channels_x,
                                                        std::vector<HcclChannelDesc>& channels_y)
{
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc;
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc));
    if (UNLIKELY(subCommRanks_.size() < AXIS_NUM_2)){
        HCCL_ERROR("[CcuTempReduceMeshMem2Mem2D::PartitionChannels] subCommRanks.size() [%d] < 2", subCommRanks_.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::vector<u32> ranks_x = subCommRanks_[0];
    std::vector<u32> ranks_y = subCommRanks_[1];

    for (u32 rankId: ranks_x) {
        if (rankId == myRank_) continue;
        auto it = rankIdToChannelDesc.find(rankId);
        if (it != rankIdToChannelDesc.end() && !it->second.empty()) {
            channels_x.push_back(it->second[0]);
        } else {
            HCCL_ERROR("[CcuTempReduceMeshMem2Mem2D::PartitionChannels] rank[%d]: there's no channel to rank [%d]", myRank_, rankId);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    
    for (u32 rankId: ranks_y) {
        if (rankId == myRank_) continue;
        auto it = rankIdToChannelDesc.find(rankId);
        if (it != rankIdToChannelDesc.end() && !it->second.empty()) {
            channels_y.push_back(it->second[0]);
        } else {
            HCCL_ERROR("[CcuTempReduceMeshMem2Mem2D::PartitionChannels] rank[%d]: there's no channel to rank [%d]", myRank_, rankId);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMeshMem2Mem2D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(2);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempReduceMeshMem2Mem2D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    uint32_t dieNum = subCommRanks_.size();
    if (dieNum != 2) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        HCCL_ERROR("[CcuTempReduceMeshMem2Mem2D] Rank[%d], Invalid IODieNum[%zu].", myRankId_, subCommRanks_.size());
        return HcclResult::HCCL_E_PARA;
    }

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh2D(comm, param, topoInfo, subCommRanks_, channelDescs));
    std::vector<HcclChannelDesc> channels_x, channels_y;
    // 划分x轴和y轴的channel
    CHK_RET(PartitionChannels(channelDescs, channels_x, channels_y));

    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要执行两次
        CcuKernelInfo kernelInfo;
    
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                return std::make_unique<CcuKernelReduceMesh2DMem2Mem>(arg);
                            };
        
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgReduceMeshMem2Mem2D>(dimSize_,
                                                                                 myRankId_,
                                                                                 rootId_,
                                                                                 axisId,
                                                                                 param,
                                                                                 subCommRanks_);
        kernelInfo.channels = axisId == 0? channels_x: channels_y;
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);

        HCCL_DEBUG("[CcuTempReduceMeshMem2Mem2D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               kernelInfo.channels.size(), subCommRanks_[axisId].size(), resourceRequest.ccuKernelInfos.size());
    }
    
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMeshMem2Mem2D::KernelRun(const OpParam& param,
                                                 const TemplateDataParams& templateDataParams,
                                                 const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t sliceSize          = templateDataParams.sliceSize;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                              static_cast<uint64_t>(buffInfo_.inputSize));

    // 前流同步
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要执行两次
        uint64_t sliceCount = sliceSize / DataTypeSizeGet(dataType_);
        uint64_t xAxisSize = (sliceCount / (dimSize_[axisId] + dimSize_[1 - axisId])) * 
                             dimSize_[0] * DataTypeSizeGet(dataType_);
        uint64_t yAxisSize = sliceSize - xAxisSize;

        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgReduceMeshMem2Mem2D>(
        inputAddr, outputAddr, sliceSize, xAxisSize, yAxisSize, token);

        HCCL_INFO("[CcuTempReduceMeshMem2Mem2D] Run Init: myRank_[%d],  inputAddr[%llu],"
                  "outputAddr[%llu], sliceSize[%llu], xAxisSize[%llu], yAxisSize[%llu], axisId_[%u]",
                  myRank_, inputAddr, outputAddr, sliceSize, xAxisSize, yAxisSize, axisId);

        void* taskArgPtr = static_cast<void*>(taskArg.get());

        HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId], templateResource.ccuKernels[axisId], taskArgPtr);
    }

    // 后流同步
    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_DEBUG("[CcuTempReduceMeshMem2Mem2D::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl 