/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "reduce_scatter_birs_executor.h"
#include "topo_experimental.h"

namespace ops_hccl_experimental {

ReduceScatterBIRSExecutor::ReduceScatterBIRSExecutor() : ReduceScatterExecutorBase()
{
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING};
}

HcclResult ReduceScatterBIRSExecutor::CalcResRequest(
    HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    if (topoInfo->serverNum == 1) {
        CHK_RET(CalcGeneralTopoInfoForA3(comm, topoInfo, algHierarchyInfo));
    } else {
        CHK_RET(CalcGeneralTopoInfoInterServer(comm, topoInfo, algHierarchyInfo));
    }
    CHK_RET(RefreshAlgType(algType));
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;

    u32 threadNum = 3;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // level0 channel
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcLevel0ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO(
        "[ReduceScatterBIRSExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level0Channels[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread.size(),
        resourceRequest.notifyNumOnMainThread, level0Channels.size());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[ReduceScatterBIRSExecutor][KernelRun] starts.");

    CHK_RET(KernelRunLevel0(param, execMem));

    HCCL_INFO("reduce scatter BIRS run success");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSExecutor::SelectAndPrepareBirsTemplate(
    bool isSingleServer, u32 localRank, u32 localRankSize, std::unique_ptr<AlgTemplateBase>& templatePtr)
{
    TemplateType tplType = isSingleServer ? TEMPLATE_REDUCE_SCATTER_BIRS : TEMPLATE_REDUCE_SCATTER_BIRS_INTER;
    templatePtr = AlgTemplateRegistry::Instance().GetAlgTemplate(tplType);
    CHK_SMART_PTR_NULL(templatePtr);
    if (isSingleServer) {
        CHK_RET(templatePtr->Prepare(localRank, localRankSize));
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_SCATTER_BIRS in COMM_LEVEL0", __func__);
    } else {
        CHK_RET(templatePtr->Prepare(topoInfo_->serverNum, topoInfo_->userRankSize));
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_SCATTER_BIRS_INTER in COMM_LEVEL0", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSExecutor::KernelRunLevel0(const OpParam& param, ExecMem& execMem)
{
    SubCommInfo level0CommInfo;
    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo));
    u32 level0LocalRank = level0CommInfo.localRank;
    u32 level0LocalRankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0LocalRank;
    u32 sliceNum = level0LocalRankSize;
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSlice(execMem.count, unitSize_, sliceNum, dataSegsSlice));

    std::unique_ptr<AlgTemplateBase> level0TempAlg = std::make_unique<AlgTemplateBaseExperimental>();
    bool isSingleServer = (topoInfo_->serverNum == 1);
    CHK_RET(SelectAndPrepareBirsTemplate(isSingleServer, level0LocalRank, level0LocalRankSize, level0TempAlg));

    HcclMem UsrInputMem{HCCL_MEM_TYPE_DEVICE, execMem.inputPtr, execMem.count * unitSize_};
    HcclMem UsrOutputMem{HCCL_MEM_TYPE_DEVICE, execMem.outputPtr, execMem.count * unitSize_};

    if (auto exp = dynamic_cast<AlgTemplateBaseExperimental*>(level0TempAlg.get())) {
        CHK_RET(exp->Prepare(
            UsrInputMem, UsrOutputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, thread_, slaveThreads_,
            param.reduceType, 0, dataSegsSlice, 0, false));

        CHK_RET(exp->RunAsync(level0LocalRank, level0LocalRankSize, channels_[COMM_LEVEL0]));
    }

    return HCCL_SUCCESS;
}

// 与仓内 experimental 示例一致：REGISTER_EXEC 需在 ops_hccl 命名空间内注册，
// 执行器类型用 ops_hccl_experimental 全限定名，避免未限定查找失败。
} // namespace ops_hccl_experimental

namespace ops_hccl {
REGISTER_EXEC("ReduceScatterBIRSExecutor", ReduceScatterBIRS, ops_hccl_experimental::ReduceScatterBIRSExecutor);
} // namespace ops_hccl
