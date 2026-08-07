/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_executor_base.h"

namespace ops_hccl_experimental {
using ops_hccl::ExecMem;
using ops_hccl::HCCL_INTERNODE_MAX_DATA_RATE;
using ops_hccl::RDMA_SEND_MAX_SIZE;
using ops_hccl::SDMA_SEND_MAX_SIZE;

ReduceScatterExecutorBase::ReduceScatterExecutorBase() : ExecutorBase() {}

// 执行入口
HcclResult ReduceScatterExecutorBase::Orchestrate(const OpParam& param, AlgResourceCtx* resCtx)
{
    HcclUs startut = TIME_NOW();
    topoInfo_ = &(resCtx->topoInfo);
    algResource_ = resCtx;
    tag_ = std::string(param.tag);
    algType_ = resCtx->algType;
    unitSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    // 做参数的还原
    ThreadHandle* threadHandlePtr
        = reinterpret_cast<ThreadHandle*>(reinterpret_cast<char*>(algResource_) + sizeof(AlgResourceCtx));
    ChannelInfo* channelInfoPtr = reinterpret_cast<ChannelInfo*>(
        reinterpret_cast<char*>(threadHandlePtr) + sizeof(ThreadHandle) * (algResource_->slaveThreadNum + 1));

    HCCL_DEBUG("[ReduceScatterExecutorBase][Orchestrate] slaveThreadNum[%u]", algResource_->slaveThreadNum);
    for (u32 i = 0; i < algResource_->slaveThreadNum + 1; i++) {
        HCCL_DEBUG("[ReduceScatterExecutorBase][Orchestrate] threadHandle[%u]=[%llu]", i, threadHandlePtr[i]);
        if (i == 0) {
            thread_ = threadHandlePtr[i];
        } else {
            slaveThreads_.push_back(threadHandlePtr[i]);
        }
    }
    AlgHierarchyInfo& algHierarchyInfo = resCtx->algHierarchyInfo;
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
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceScatterExecutorBase][Orchestrate]errNo[0x%016llx] ReduceScatter executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO(
        "[ReduceScatterExecutorBase][Orchestrate]tag[%s] ReduceScatter executor orchestrate success, take time "
        "[%lld]us.",
        param.tag, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

bool ReduceScatterExecutorBase::IsHugeData(u64 curSize) const
{
    bool hugeData = curSize * topoInfo_->userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult ReduceScatterExecutorBase::RunLoop(const OpParam& param)
{
    u64 totalRecvCount = param.DataDes.count;
    u64 totalRecvSize = totalRecvCount * unitSize_;

    u8* curUserInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curUserOutputPtr = static_cast<u8*>(param.outputPtr);
    auto cclInputMem = algResource_->cclInputMem;
    auto cclOutputMem = algResource_->cclOutputMem;
    CHK_PRT_RET(
        (cclInputMem.size == 0), HCCL_ERROR("[ReduceScatterExecutorBase][RunLoop]cclBuffer size is zero"), HCCL_E_PARA);

    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommAcquireComm(param.commName);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] [%s] HcommAcquireComm failed ", __func__, param.commName),
            static_cast<HcclResult>(ret));
    }

    u64 curRecvCount = totalRecvCount;
    u64 curRecvSize = curRecvCount * unitSize_;
    u64 curSendSize = topoInfo_->userRankSize * curRecvSize;

#ifndef AICPU_COMPILE
    if (!IsHugeData(curRecvSize)) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
    }
#endif
    HcclMem curInputMem{cclInputMem.type, cclInputMem.addr, curSendSize};
    HcclMem curOutputMem{cclOutputMem.type, cclOutputMem.addr, curRecvSize};

    ExecMem execMem;
    execMem.count = curRecvCount;
    execMem.inputMem = curInputMem;
    execMem.outputMem = curOutputMem;
    execMem.inputPtr = curUserInputPtr;
    execMem.outputPtr = curUserOutputPtr;

    HCCL_DEBUG(
        "[ReduceScatterExecutorBase][RunLoop] curUserInputPtr[%p], curUserOutputPtr[%p], "
        "curRecvCount[%llu], curRecvSize[%llu], curSendSize[%llu], inputPtr[%p], outputPtr[%p]",
        curUserInputPtr, curUserOutputPtr, curRecvCount, curRecvSize, curSendSize, curInputMem.addr, curOutputMem.addr);

    CHK_RET(KernelRun(param, execMem));

#ifndef AICPU_COMPILE
    if (!IsHugeData(curRecvSize)) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    }
#endif
    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommReleaseComm(param.commName);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] [%s] HcommReleaseComm failed ", __func__, param.commName),
            static_cast<HcclResult>(ret));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterExecutorBase::CalcResRequest(
    HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterExecutorBase::PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice)
{
    CHK_PRT_RET(
        (sliceNum == 0), HCCL_ERROR("[ReduceScatterExecutorBase][PrepareDataSlice]sliceNum is zero."), HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        dataSlice[i].offset = (i * sliceSize);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_experimental
