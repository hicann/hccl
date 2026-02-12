/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_reduce_scatter_v_mesh_1D.h"

namespace ops_hccl {
InsTempReduceScatterVMesh1D::InsTempReduceScatterVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempReduceScatterVMesh1D::~InsTempReduceScatterVMesh1D()
{
}

HcclResult InsTempReduceScatterVMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                                               AlgResourceRequest& resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterVMesh1D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    dataType_ = param.vDataDes.dataType;
    allRankProcessSize_ = tempAlgParams.allRankSliceSize;
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    allRankCounts_.resize(templateRankSize_);
    for (u64 i = 0; i < allRankProcessSize_.size(); ++i) {
        allRankCounts_[i] = allRankProcessSize_[i] / dataTypeSize; 
    }
    HCCL_INFO("[InsTempReduceScatterVMesh1D] Run Start");
    if (threadNum_ > 1) {
        // 定义在alg_v2_template_base.h中的基类
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunReduceScatterV(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        // 定义在alg_v2_template_base.h中的基类
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    PostCopy(tempAlgParams, templateResource.threads);
    HCCL_INFO("[InsTempReduceScatterVMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVMesh1D::PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    // 通信结束之后，数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterVMesh1D][RunReduceScatterV] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 cclBuffer 拷贝到 userOut
    HCCL_INFO("[InsTempReduceScatterVMesh1D][PostCopy], copy from cclBuffer to userOut");
    // 先把本卡的数据从userIn搬运到userOut，然后再在userOut上做规约
    HCCL_INFO("[InsTempReduceScatterVMesh1D][PostCopy]tempAlgParams.repeatNum=%llu", tempAlgParams.repeatNum);
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                                       tempAlgParams.buffInfo.inBuffBaseOff +
                                       repeatIdx * tempAlgParams.inputRepeatStride +
                                       tempAlgParams.allRankDispls[myAlgRank],
                                       allRankProcessSize_[myAlgRank], allRankCounts_[myAlgRank]);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
                                       tempAlgParams.buffInfo.outBuffBaseOff +
                                       repeatIdx * tempAlgParams.outputRepeatStride,
                                       allRankProcessSize_[myAlgRank], allRankCounts_[myAlgRank]);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        
        // 把其他卡的数据input累加到output
        for (u32 tmpRank = 0; tmpRank < templateRankSize_; tmpRank++) {
            if (tmpRank != myAlgRank) {
                // 搬的数据量大小为当前卡接收数据量  
                DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                               tempAlgParams.buffInfo.hcclBuffBaseOff
                                               + repeatIdx * tempAlgParams.outputRepeatStride
                                               + tmpRank * tempAlgParams.outputSliceStride,
                                               allRankProcessSize_[myAlgRank], allRankCounts_[myAlgRank]);
                DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
                                               tempAlgParams.buffInfo.outBuffBaseOff
                                               + repeatIdx * tempAlgParams.outputRepeatStride,
                                               allRankProcessSize_[myAlgRank], allRankCounts_[myAlgRank]);
                CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVMesh1D::RunReduceScatterV(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParam)
{
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterVMesh1D][RunReduceScatterV] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    for (size_t i = 0; i < allRankProcessSize_.size(); ++i) {
    HCCL_INFO("Rank %zu has process size: %llu", i, allRankProcessSize_[i]);
}
    for (u32 queIdx = 0; queIdx < threadNum_; queIdx++) {
        u32 nextRank = (myAlgRank + 1 + queIdx) % templateRankSize_; // 这里取的虚拟rankId
        u32 remoteRank = subCommRanks_[0][nextRank];
        HCCL_DEBUG("[InsTempReduceScatterVMesh1D][RunReduceScatterV] myRank[%d], toRank[%d], fromRank[%d]",
                   myRank_, remoteRank, remoteRank);
        const ChannelInfo &linkSend = channels.at(remoteRank)[0];
        const ChannelInfo &linkRecv = channels.at(remoteRank)[0];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        // 在 HcclBuffer 上进行 ReduceScatterV 操作
        // 由于进程只能访问远端的HcclBuffer，所以只能通过write的方式将自己userIn上的数据写到远端HcclBuffer上
        for (u32 repeatIdx = 0; repeatIdx < tempAlgParam.repeatNum; repeatIdx++) {
            HCCL_INFO("[InsTempReduceScatterVMesh1D] start5");
            // 在reduce_scatter_v_op.cc的创建channels的环节中获取到了remote的HcclBuff的地址
            void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
            // 在接收的时候接收源应该是远端地址，但是由于rs的mesh算法用的是write，所以rx不用care
            DataSlice txSrcSlice = DataSlice(tempAlgParam.buffInfo.inputPtr, tempAlgParam.buffInfo.inBuffBaseOff +
                repeatIdx * tempAlgParam.inputRepeatStride + tempAlgParam.allRankDispls[nextRank],
                allRankProcessSize_[nextRank], allRankCounts_[nextRank]); // 发送源
            DataSlice txDstSlice = DataSlice(remoteCclBuffAddr, tempAlgParam.buffInfo.hcclBuffBaseOff +
                repeatIdx * tempAlgParam.outputRepeatStride + myAlgRank * tempAlgParam.outputSliceStride,
                allRankProcessSize_[nextRank], allRankCounts_[nextRank]);  // 发送目标
            
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
        }
        SendRecvInfo sendRecvInfo{{linkSend, linkRecv},
                             {{txSrcSlices, txDstSlices},{rxSrcSlices, rxDstSlices}}};

        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempReduceScatterVMesh1D] RunReduceScatterV Send failed"),
                    HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterVMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}
 
void InsTempReduceScatterVMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}
} // namespace Hccl