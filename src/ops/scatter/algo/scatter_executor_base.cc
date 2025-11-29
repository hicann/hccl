/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_executor_base.h"

namespace ops_hccl {
ScatterExecutorBase::ScatterExecutorBase() : ExecutorBase()
{
}

// 执行入口
HcclResult ScatterExecutorBase::Orchestrate(const OpParam &param, AlgResourceCtx* resCtx)
{
    HcclUs startut = TIME_NOW();
    topoInfo_ = &(resCtx->topoInfo);
    algResource_ = resCtx;
    tag_ = std::string(param.tag);
    algType_ = resCtx->algType;
    unitSize_ = SIZE_TABLE[param.DataDes.dataType];

    // 参数校验
    if (topoInfo_->userRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
    }
    CHK_PTR_NULL(param.outputPtr);

    // 做参数的还原
    ThreadHandle* threadHandlePtr = reinterpret_cast<ThreadHandle *>(reinterpret_cast<u8 *>(algResource_) + sizeof(AlgResourceCtx));

    ChannelInfo* channelInfoPtr = reinterpret_cast<ChannelInfo *>(threadHandlePtr + algResource_->slaveThreadNum + 1);
    HCCL_DEBUG("[ScatterExecutorBase][Orchestrate] slaveThreadNum[%u]", algResource_->slaveThreadNum);
    for (u32 i = 0; i < algResource_->slaveThreadNum + 1; i++) {
        HCCL_DEBUG("[ScatterExecutorBase][Orchestrate] threadHandle[%u]=[%llu]", i, threadHandlePtr[i]);
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
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ScatterExecutorBase][Orchestrate]errNo[0x%016llx]Scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("[ScatterExecutorBase][Orchestrate]tag[%s] Scatter executor orchestrate success, take time [%lld]us.",
        param.tag, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

bool ScatterExecutorBase::IsHugeData(u64 curSize)
{
    bool hugeData = curSize * topoInfo_->userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult ScatterExecutorBase::RunLoop(const OpParam &param)
{
    u32 root = param.root;
    u64 totalRecvCount = param.DataDes.count;
    u64 totalRecvSize = totalRecvCount * unitSize_;

    u8 *curUserInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curUserOutputPtr = static_cast<u8 *>(param.outputPtr);
    auto cclInputMem = algResource_->cclInputMem;
    auto cclOutputMem = algResource_->cclOutputMem;
    CHK_PRT_RET((cclInputMem.size == 0), HCCL_ERROR("[ScatterExecutorBase][RunLoop]cclBuffer size is zero"), HCCL_E_PARA);

    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclInputMem.size / topoInfo_->userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize_;
    HCCL_DEBUG("[ScatterExecutorBase][RunLoop]tag[%s], userRankSize is [%u], root is [%u], "
               "maxCountPerLoop is [%llu], totalRecvCount is [%llu]",
        tag_.c_str(), topoInfo_->userRankSize, root, maxCountPerLoop, totalRecvCount);

    for (u64 countLeft = totalRecvCount, curRecvCount = 0; countLeft > 0; countLeft -= curRecvCount) {
        curRecvCount = (countLeft > maxCountPerLoop ? maxCountPerLoop : countLeft);
        u64 curRecvSize = curRecvCount * unitSize_;
        u64 curSendSize = topoInfo_->userRankSize * curRecvSize;

#ifndef AICPU_COMPILE
        if (!IsHugeData(curRecvSize)) {
            CHK_RET(HcommSetLaunchMode(param.algTag, LAUNCH_MODE_BATCH));
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

        HCCL_DEBUG("[ScatterExecutorBase][RunLoop] curUserInputPtr[%p], curUserOutputPtr[%p], "
            "curRecvCount[%llu], curRecvSize[%llu], curSendSize[%llu], inputPtr[%p], outputPtr[%p]", curUserInputPtr,
            curUserOutputPtr, curRecvCount, curRecvSize, curSendSize, curInputMem.addr, curOutputMem.addr);

        // root节点需要将数据从userIn拷贝到cclIn上
        if (topoInfo_->userRank == root) {
            for (u32 i = 0; i < topoInfo_->userRankSize; i++) {
                void* src = (u8*)execMem.inputPtr + totalRecvSize * i;
                void* dst = (u8*)execMem.inputMem.addr + curRecvSize * i;
                CHK_RET(HcommLocalCopyOnThread(thread_, dst, src, curRecvSize));
            }
        }

        if (curRecvSize % HCCL_MIN_SLICE_ALIGN != 0) { // 不支持内存不对齐的轮次
            DMAReduceFlag_ = false;
        }

        CHK_RET(KernelRun(param, execMem));

        // 将cclOut上的数据搬运到userOut
        if (!DMAReduceFlag_) {
            void* src = execMem.outputMem.addr;
            void* dst = execMem.outputPtr;
            CHK_RET(HcommLocalCopyOnThread(thread_, dst, src, curRecvSize));
        }

        curUserInputPtr += curRecvSize;
        curUserOutputPtr += curRecvSize;

#ifndef AICPU_COMPILE
        if (!IsHugeData(curRecvSize)) {
            CHK_RET(HcommSetLaunchMode(param.algTag, LAUNCH_MODE_EAGER));
        }
#endif
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterExecutorBase::CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
    AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType)
{
    return HCCL_SUCCESS;
}

HcclResult ScatterExecutorBase::PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum,
    std::vector<Slice> &dataSlice)
{
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[ScatterExecutorBase][PrepareDataSlice]sliceNum is zero."), HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        dataSlice[i].offset = (i * sliceSize);
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterExecutorBase::KernelRunLevel1(HcclMem &inputMem, u64 count, HcclDataType dataType,
    u32 &commIndex, u32 root, u32 &subRoot, CommPlane commLevel, ThreadHandle thread)
{
    SubCommInfo subCommInfo;
    CHK_RET(GetSubCommInfo(commLevel, subCommInfo));

    u32 subCommSize = subCommInfo.localRankSize;
    if (subCommSize <= 1 || subRoot != topoInfo_->userRank) {
        HCCL_INFO("[Scatter][KernelRunLevel1]: no need to run intra-server, subCommSize[%u], subRoot[%u]," \
            "userRank[%u]", subCommSize, subRoot, topoInfo_->userRank);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[Scatter][KernelRunLevel1]: start to run intra-server, subCommSize[%u], subRoot[%u]," \
        "userRank[%u]", subCommSize, subRoot, topoInfo_->userRank);

    u32 rootRankLevel1 = 0;
    CHK_RET(GetSubCommRankByUserRank(root, commLevel, algResource_->algHierarchyInfo, rootRankLevel1));

    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        // server间NB算法走NB
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NB);
        HCCL_INFO("[ScatterExecutorBase][KernelRunLevel1] using NB algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NHR);
        if (topoInfo_->deviceType != DevType::DEV_TYPE_910_93) {
            level1TempAlg->CloseBarrier();
        }
        HCCL_INFO("[ScatterExecutorBase][KernelRunLevel1] using NHR algo inter-server.");
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING);
        HCCL_INFO("[ScatterExecutorBase][KernelRunLevel1] using ring algo inter-server.");
    }

    CHK_SMART_PTR_NULL(level1TempAlg);
    CHK_RET(level1TempAlg->Prepare(inputMem, inputMem, inputMem, count * topoInfo_->userRankSize,
        dataType, thread_, HCCL_REDUCE_RESERVED, rootRankLevel1, std::vector<Slice>(0))); // count是output的数据个数
    CHK_RET(level1TempAlg->RunAsync(subCommInfo.localRank, subCommInfo.localRankSize, channels_[commLevel]));
    return HCCL_SUCCESS;
}

// 计算userRank所在平面的subRoot rank编号
void GetSubRootRank(u32 root, u32 curLevel, AlgHierarchyInfo& algHierarchyInfo, u32 &subRootRank)
{
    u32 preLevelsRankSize = 1;
    subRootRank = 0;
    for (u32 level = 0; level < algHierarchyInfo.levels; level++) {
        if (level == curLevel) {
            subRootRank += algHierarchyInfo.infos[curLevel].localRank * preLevelsRankSize;
        } else {
            u32 subCommRank = root / preLevelsRankSize % algHierarchyInfo.infos[level].localRankSize;
            subRootRank += subCommRank * preLevelsRankSize;
        }
        preLevelsRankSize *= algHierarchyInfo.infos[level].localRankSize;
    }
    HCCL_INFO("[GetSubRootRank]root[%u] level[%u] -> subRootRank[%u]", root, curLevel, subRootRank);
    return;
}

}
