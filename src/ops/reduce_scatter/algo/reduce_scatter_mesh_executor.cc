/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_executor.h"
#include "hcomm_primitives.h"
#include <unistd.h>

namespace ops_hccl {

ReduceScatterMeshExecutor::ReduceScatterMeshExecutor() : ExecutorBase()
{
}

HcclResult ReduceScatterMeshExecutor::Orchestrate(const OpParam &param, AlgResourceCtx* resCtx)
{
    HcclUs startut = TIME_NOW();
    topoInfo_ = &(resCtx->topoInfo);
    algResource_ = resCtx;
    tag_ = std::string(param.tag);
    algType_ = resCtx->algType;
    unitSize_ = SIZE_TABLE[param.DataDes.dataType];

    // 参数校验
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    // 做参数的还原
    ThreadHandle* threadHandlePtr = reinterpret_cast<ThreadHandle *>(reinterpret_cast<u8 *>(algResource_) +
        sizeof(AlgResourceCtx));
    ChannelInfo* channelInfoPtr = reinterpret_cast<ChannelInfo *>(threadHandlePtr + algResource_->slaveThreadNum + 1);
    for (u32 i = 0; i < algResource_->slaveThreadNum + 1; i++) {
        if (i == 0) {
            thread_ = threadHandlePtr[i];
        } else {
            slaveThreads_.push_back(threadHandlePtr[i]);
        }
    }
    AlgHierarchyInfo algHierarchyInfo = resCtx->algHierarchyInfo;
    channels_.resize(algHierarchyInfo.levels);
    for (u32 level = 0; level < algHierarchyInfo.levels; level++) {
        u32 curLevelRankSize = algHierarchyInfo.infos[level].localRankSize;
        channels_[level].resize(curLevelRankSize);
        for (u32 rank = 0; rank < curLevelRankSize; rank++) {
            channels_[level][rank] = channelInfoPtr[rank];
        }
        channelInfoPtr += curLevelRankSize;
    }

    HcclResult ret = RunLoop(param);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterMeshExecutor][Orchestrate]errNo[0x%016llx] Reduce scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("[ReduceScatterMeshExecutor][Orchestrate]tag[%s] Reduce scatter executor orchestrate success, take time [%lld]us.",
        param.tag, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshExecutor::CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
    AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType)
{
    CHK_RET(CalcGeneralTopoInfoForCommHostDpu(topoInfo, algHierarchyInfo));
    // AICPU资源
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread = 0;

    u32 level1RankSize = algHierarchyInfo.infos[1].localRankSize;
    std::vector<HcclChannelDesc> level0Channels;
    resourceRequest.channels.push_back(level0Channels);
    // host网卡处理框间的数据传送，全连接
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcLevel1ChannelRequestHostDpu(comm, param, topoInfo, algHierarchyInfo, algType, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[ScatterRingExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level1Channels.size());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshExecutor::GetSubCommInfoA5(const CommPlane levelIndex, SubCommInfo &info)
{
    AlgHierarchyInfo& algHierarchyInfo = algResource_->algHierarchyInfo;
    if (levelIndex >= algHierarchyInfo.levels) {
        HCCL_ERROR("[ReduceScatterMeshExecutor][GetSubCommInfo]tag[%s], levelIndex[%u] exceeds actual levels[%u]",
            tag_.c_str(), levelIndex, algHierarchyInfo.levels);
        return HCCL_E_INTERNAL;
    }
    info.localRank = algHierarchyInfo.infos[levelIndex].localRank;
    info.localRankSize = algHierarchyInfo.infos[levelIndex].localRankSize;
    return HCCL_SUCCESS;
}


HcclResult ReduceScatterMeshExecutor::RunLoop(const OpParam &param)
{
// #ifdef AICPU_COMPILE
    SubCommInfo info;
    GetSubCommInfoA5(COMM_LEVEL1, info);

    u64 outputCount = param.DataDes.count;
    u64 outputSize = outputCount * unitSize_;

    u64 localRankSize = info.localRankSize;
    CHK_PRT_RET((localRankSize == 0), HCCL_ERROR("[ReduceScatterMeshExecutor][RunLoop]rank size is zero"), HCCL_E_PARA);
    // buffer大小是接口提供的，直接使用 然后按照Buffer大小切分 执行template
    auto cclInputMem = algResource_->cclInputMem;
    auto cclOutputMem = algResource_->cclOutputMem;
    CHK_PRT_RET((cclInputMem.size == 0), HCCL_ERROR("[ReduceScatterMeshExecutor][RunLoop]cclBuffer size is zero"), HCCL_E_PARA);
    u64 maxLoopOutputCount = cclInputMem.size / (localRankSize * unitSize_);
    u64 loopTimes          = (outputCount + maxLoopOutputCount - 1) / maxLoopOutputCount;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 loopOffsetCount = loop * maxLoopOutputCount;
        u64 loopOffsetSize  = loopOffsetCount * unitSize_;
        // 本轮需要处理的 output 数据总量
        u64 currOutputCount = (loop == (loopTimes - 1)) ? outputCount - loopOffsetCount : maxLoopOutputCount;
        u64 currSliceSize   = currOutputCount * unitSize_;

        // 搬运需要参数计算
        for (u32 rankId = 0; rankId < localRankSize; rankId++) {
            // 将本端数据从sendbuf写入到CCL IN，用来后续写入到对端
            void *src = static_cast<u8 *>(param.inputPtr) + outputSize * rankId + loopOffsetSize;
            void *dst = static_cast<u8 *>(cclInputMem.addr) + currSliceSize * rankId;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, dst, src, currSliceSize)));
            HCCL_INFO("local copy from sendbuf {%llu} to inputmem {%llu}, size: {%llu}",
                outputSize * rankId + loopOffsetSize,
                currSliceSize * rankId,
                currSliceSize);
        }

        // 将本卡当前循环的数据，从sendbuf搬运到CCL-IN，用于后续进行
        void *src = static_cast<u8 *>(param.inputPtr) + outputSize * info.localRank + loopOffsetSize;
        void *dst = static_cast<u8 *>(cclOutputMem.addr) + currSliceSize * info.localRank;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, dst, src, currSliceSize)));
        algResource_->dpuResCtx.sliceSize = currSliceSize;

        // set mode - eager
        if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }

        if (HcommThreadSynchronize(thread_) != 0) {
            HCCL_ERROR("HcommThreadSynchronize failed");
            return HCCL_E_INTERNAL;
        }
        // AICPU - DPU算法前同步
        void *npu2DpuShmemPtr = algResource_->npu2DpuShmemPtr;

        // 向DPU下发数据
        u32 sendMsgId = 0;
        if (HcommSendRequest(reinterpret_cast<uint64_t>(npu2DpuShmemPtr), param.algTag,
            static_cast<void*>(&algResource_->dpuResCtx), sizeof(DPUAlgResourceCtx), &sendMsgId) != 0) {
            HCCL_ERROR("HcommSendRequest failed");
            return HCCL_E_INTERNAL;
        }
        // 等待DPU数据传输，然后回写结果回来
        void* dpu2NpuShmemPtr = algResource_->dpu2NpuShmemPtr;
        void *recvData = nullptr;
        u32 recvMsgId = 0;

        if (HcommWaitResponse(reinterpret_cast<uint64_t>(dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
            HCCL_ERROR("HcommWaitResponse failed");
            return HCCL_E_INTERNAL;
        }

        // set mode - batch
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }

        if (recvMsgId != sendMsgId) {
            HCCL_ERROR("recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
            return HCCL_E_INTERNAL;
        }

        // CCL_IN上面做规约，输出到CCL_OUT上
        std::unique_ptr<AlgTemplateBase> localReduceTemp;
        localReduceTemp = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCE_SCATTER_LOCAL_REDUCE);
        void* loopOutputPtr = static_cast<s8 *>(param.outputPtr) + loopOffsetSize;
        HcclMem outputMem = {HCCL_MEM_TYPE_DEVICE, loopOutputPtr, outputSize};
        localReduceTemp->Prepare(
            cclOutputMem, // CCL IN
            outputMem, // 规约到recvBuf上
            cclOutputMem,
            currOutputCount,
            param.DataDes.dataType,
            thread_,
            param.reduceType,
            0, {{loopOffsetSize, currSliceSize}});

        // AICPU算法展开，进行local reduce
        CHK_RET(localReduceTemp->RunAsync(info.localRank, info.localRankSize, channels_[1]));
    }
// #endif
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterMeshExecutor", ReduceScatterLocalReduceTemplate, ReduceScatterMeshExecutor);
}