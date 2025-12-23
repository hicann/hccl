/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_comm_executor.h"
#include "config_log.h"

namespace ops_hccl {
ScatterCommExecutor::ScatterCommExecutor() : ScatterExecutorBase()
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

HcclResult ScatterCommExecutor::CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
    AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType)
{
    CHK_RET(CalcGeneralTopoInfoForComm(comm, topoInfo, algHierarchyInfo));
    CHK_RET(RefreshAlgType(algType));

    u32 level0RankSize = algHierarchyInfo.infos[COMM_LEVEL0].localRankSize;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;

    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread = 1;
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // level0 channel
    std::vector<HcclChannelDesc> level0Channels;
    resourceRequest.channels.push_back(level0Channels);

    // level1 channel
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcLevel1ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    HCCL_INFO("[ScatterRingExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level0Channels[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level0Channels.size(), level1Channels.size());
    return HCCL_SUCCESS;
}

HcclResult ScatterCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[ScatterCommExecutor] scatter starts.");
    HcclMem &inputMem = execMem.inputMem;
    HcclMem &outputMem = execMem.outputMem;
    u64 count = execMem.count;
    auto root = param.root;
    auto dataType = param.DataDes.dataType;
    u32 userRank = topoInfo_->userRank;

    u32 commIndex = 0;
    // 统一走server间
    SubCommInfo combinedCommInfo;
    CHK_RET(GetSubCommInfo(COMM_LEVEL1, combinedCommInfo));
    // commIndex的作用是什么？
    CHK_RET(KernelRunLevel1(inputMem, count, dataType, commIndex, root, userRank, COMM_LEVEL1, thread_));

    // 从CCL_IN拷贝到CCL_OUT
    u8* src = static_cast<u8 *>(inputMem.addr) + outputMem.size * combinedCommInfo.localRank;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem.addr, src, outputMem.size)));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterCommExecutor", ScatterComm, ScatterCommExecutor);

}
