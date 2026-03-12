/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OP_COMMON
#define OPS_HCCL_OP_COMMON

#include <string>
#include <memory>
#include "hccl.h"

#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

namespace ops_hccl {

HcclResult HcclExecOp(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, std::string &algName);

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo);

HcclResult HcclGetAlgRes(HcclComm comm, OpParam &param, std::shared_ptr<InsCollAlgBase> &executor, TopoInfoWithNetLayerDetails *topoInfo,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void **resCtxSequence, bool &isResourceReused);

HcclResult GetAlgResAICPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag);

HcclResult HcclAllocAlgResourceAICPU(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclGetThread(HcclComm comm, const OpParam &param,
                        AlgResourceRequest &resRequest, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclGetChannel(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclGetCcuKernel(HcclComm comm, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclGetChannelForCcu(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest);

HcclResult HcclAllocAlgResourceCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                                   std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);
HcclResult GetAlgResCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                        std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
                        AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize);

HcclResult GetAlgResAiv(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence);

HcclResult HcclAllocAlgResourceAiv(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, AlgResourceCtxSerializable* resCtxHost);

HcclResult GetAlgResDPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag);

HcclResult CheckCount(const u64 count);

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);

std::string GetSupportDataType(bool needReduce);

HcclResult SetCommEngine(OpParam &param);

void CompReqChannelWithExistChannel(const std::vector<std::vector<ChannelInfo>>& existChannels,
                                    AlgResourceRequest &resRequest);

HcclResult HcclMemcpyCtxHostToDevice(HcclComm comm, const OpParam &param,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void **resCtxSequence, uint64_t& ctxSize);

bool CheckHCCLIndependentOp();

HcclResult SingleRankProc(const OpParam &param);

HcclResult HcclCheckTag(const char *tag);

HcclResult SetOpParamAlgTag(OpParam &param, const std::string &algName);

HcclResult SaveMainThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle thread, u32 notifyNum);

HcclResult GetMainThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle &thread, u32 &notifyNum);

HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
                         std::string &algName);

HcclResult HcclAicpuKernelEntranceLaunch(HcclComm comm, OpParam &param, ThreadHandle cpuTsThread,
    ThreadHandle exportedCpuTsThread, u32 notifyNumOnMainThread, void *resCtxSequence, std::string &algName);

HcclResult AicpuKernelLaunch(OpParam &param);

HcclResult HcclAivKernelEntranceLaunch(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    AlgResourceCtxSerializable &resCtxHost);

HcclResult HcclGetOpExpansionMode(HcclComm comm, OpParam &param);

bool HcclCheckAicpuEnableOpen();

}  // namespace ops_hccl

#endif