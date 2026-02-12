/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_mesh_executor.h"

namespace ops_hccl {
ScatterMeshExecutor::ScatterMeshExecutor() : ScatterExecutorBase()
{
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_NHR,
        AlgTypeLevel1::ALG_LEVEL1_NB,
        AlgTypeLevel1::ALG_LEVEL1_RING
    };
    desc_.level2SupportedAlgos = {
        AlgTypeLevel2::ALG_LEVEL2_NHR,
        AlgTypeLevel2::ALG_LEVEL2_NB,
        AlgTypeLevel2::ALG_LEVEL2_RING
    };
}

HcclResult ScatterMeshExecutor::CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
    AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType)
{
    CHK_RET(CalcGeneralTopoInfoForA2(comm, topoInfo, algHierarchyInfo));
    CHK_RET(RefreshAlgType(algType));

    u32 level0RankSize = algHierarchyInfo.infos[COMM_LEVEL0].localRankSize;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;  // 主thread可以通过接口传入的stream来做转换
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcLevel0ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcLevel1ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    HCCL_INFO("[ScatterRingExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level0Channels[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread.size(), resourceRequest.notifyNumOnMainThread,
        level0Channels.size(), level1Channels.size());
    return HCCL_SUCCESS;
}

HcclResult ScatterMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[ScatterMeshExecutor] scatter starts.");

    SubCommInfo level0CommInfo;
    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo));
    u32 level0LocalRank = level0CommInfo.localRank;
    u32 level0LocalRankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0LocalRank;

    SubCommInfo level1CommInfo;
    CHK_RET(GetSubCommInfo(COMM_LEVEL1, level1CommInfo));
    u32 level1LocalRank = level1CommInfo.localRank;
    u32 level1LocalRankSize = level1CommInfo.localRankSize;

    bool bRet = level0LocalRankSize == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ScatterMeshExecutor][KernelRun]tag[%s],comm level0 is empty", tag_.c_str()),
        HCCL_E_INTERNAL);

    /* ***********第一步: 节点间scatter ****************************/
    u32 subRoot = INVALID_VALUE_RANKID;
    GetSubRootRank(param.root, COMM_LEVEL1, algResource_->algHierarchyInfo, subRoot);
    CHK_PRT_RET(subRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR("[ScatterMeshExecutor][KernelRun]GetSubRootForScatter failed, "\
        "userRank[%u], root[%u], subRoot[%u]", topoInfo_->userRank, param.root, subRoot), HCCL_E_INTERNAL);
    HCCL_DEBUG("[ScatterMeshExecutor][KernelRun]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        topoInfo_->userRank, param.root, subRoot);
    CHK_RET(KernelRunLevel1(execMem.inputMem, execMem.count, param.DataDes.dataType, commIndex,
        param.root, subRoot, COMM_LEVEL1, thread_));

    /* ***********第二步: 节点内scatter *****************************/
    // 根据数据量算每个环上数据的偏移和大小
    u32 sliceNum = level0LocalRankSize;
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSlice(execMem.count, unitSize_, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size / level1LocalRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * level1LocalRank;
    HcclMem scatterMeshInput = HcclMemRange(execMem.inputMem, serverSliceOffset, serverSliceSize);
    HcclMem scatterMeshOutput = HcclMemRange(execMem.inputMem, serverSliceOffset, serverSliceSize);

    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_SCATTER_MESH);
    CHK_SMART_PTR_NULL(level0TempAlg);
    // 这个prepare接口可以优化掉
    CHK_RET(level0TempAlg->Prepare(level0LocalRank, level0LocalRankSize));
    // 偏移需要带入prepare
    u32 rootRankLevel0 = 0;
    CHK_RET(GetSubCommRankByUserRank(subRoot, COMM_LEVEL0, algResource_->algHierarchyInfo, rootRankLevel0));
    CHK_PRT_RET(rootRankLevel0 == INVALID_VALUE_RANKID,
        HCCL_ERROR("[ScatterMeshExecutor][KernelRun]rootRankLevel0[%u] is invalid, userRank[%u], subRoot[%u]",
        rootRankLevel0, topoInfo_->userRank, subRoot), HCCL_E_INTERNAL);

    CHK_RET(level0TempAlg->Prepare(scatterMeshInput, scatterMeshOutput, execMem.inputMem, execMem.count,
        param.DataDes.dataType, thread_, HCCL_REDUCE_RESERVED, rootRankLevel0, dataSegsSlice, serverSliceOffset));

    CHK_RET(level0TempAlg->RunAsync(level0LocalRank, level0LocalRankSize, channels_[COMM_LEVEL0]));

    u8* src = static_cast<u8 *>(scatterMeshOutput.addr) + execMem.outputMem.size * level0LocalRank;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, execMem.outputMem.addr, src, execMem.outputMem.size)));
    return HCCL_SUCCESS;
}

// 注册executor实现，第一个字符串为算法名字，第三个参数为实现类
REGISTER_EXEC("ScatterMeshExecutor", ScatterMesh, ScatterMeshExecutor);

}
