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

HcclResult HcclExecOp(HcclComm comm, OpParam &param);

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, TopoInfo **topoInfo);

HcclResult HcclGetAlgRes(HcclComm comm, OpParam &param, std::shared_ptr<InsCollAlgBase> &executor, TopoInfo *topoInfo,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void **resCtxSequence, bool &isResourceReused);

HcclResult GetAlgResAICPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag);

HcclResult HcclAllocAlgResourceAICPU(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

HcclResult HcclGetH2DNotify(std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

HcclResult HcclGetThread(HcclComm comm, const OpParam &param,
                        AlgResourceRequest &resRequest, std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

HcclResult HcclGetChannel(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxSequenceHost);

HcclResult HcclGetCcuKernel(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclGetChannelForCcu(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);

HcclResult HcclAllocAlgResourceCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                                   std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost);
HcclResult GetAlgResCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                        std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo* topoInfo,
                        AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize);

HcclResult GetAlgResAiv(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize);

HcclResult HcclAllocAlgResourceAiv(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, AlgResourceCtxSerializable* resCtxHost);

HcclResult GetAlgResDPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag);

HcclResult CheckCount(const u64 count);

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);

std::string GetSupportDataType(bool needReduce);

HcclResult SetCommEngine(OpParam &param, OpExecuteConfig opExecuteConfig);

void CompReqChannelWithExistChannel(const std::vector<std::vector<ChannelInfo>>& existChannels,
                                    AlgResourceRequest &resRequest);

HcclResult HcclMemcpyCtxHostToDevice(HcclComm comm, const OpParam &param,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void **resCtxSequence, uint64_t& ctxSize);

bool CheckHCCLIndependentOp();

HcclResult SingleRankProc(const OpParam &param);

HcclResult HcclCheckTag(const char *tag);

HcclResult SetOpParamAlgTag(OpParam &param, const std::string &algName);
}  // namespace ops_hccl

#endif