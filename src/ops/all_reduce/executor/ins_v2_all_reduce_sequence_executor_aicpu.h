/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_EXECUTOR_AICPU_H
#define HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_EXECUTOR_AICPU_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2, typename InsAlgTemplate3>
class InsV2AllReduceSequenceExecutorAicpu : public InsCollAlgBase {
public:
    explicit InsV2AllReduceSequenceExecutorAicpu();
    ~InsV2AllReduceSequenceExecutorAicpu() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

#ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam &param, const CcuFastLaunchCtx *resCtx) override;
    HcclResult FastLaunchSaveCtx(const OpParam &param, const TemplateResource &templateAlgResStepOne,
                                 const TemplateResource &templateAlgResStepTwo,
                                 const TemplateResource &templateAlgResStepThree,
                                 const TemplateResource &templateAlgResStepFour, u32 notifyNumOnMainThread);
#endif

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    void GenBaseTempAlgParams(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        TemplateDataParams &tempAlgParamsStepOne, TemplateDataParams &tempAlgParamsStepTwo,
        TemplateDataParams &tempAlgParamsStepThree, TemplateDataParams &tempAlgParamsStepFour) const;
    void GenTempAlgParamsStepOne(const u64 loop, const u64 currDataCount, const u64 processedDataCount,
        TemplateDataParams &tempAlgParamsStepOne) const;
    void GenTempAlgParamsStepTwo(const u64 loop, const u64 currDataCount, const u64 sliceSizeLastStep,
        const u64 tailSizeLastStep, TemplateDataParams &tempAlgParamsStepTwo) const;
    void GenTempAlgParamsStepThree(const u64 loop, const u64 currDataCount, const u64 sliceSize,
        const u64 tailSize, TemplateDataParams &tempAlgParamsStepThree) const;
    void GenTempAlgParamsStepFour(const u64 loop, const u64 currDataCount, const u64 processedDataCount,
        const u64 sliceSize, const u64 tailSize, TemplateDataParams &tempAlgParamsStepFour) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempReousrce) const;

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    uint64_t outCclBuffSize_{0};
    uint64_t inCclBuffSize_{0};
    uint64_t outCclBuffOffset_{0};
    uint64_t inCclBuffOffset_{0};

    u64 scratchBlockSize_{0};
    CommEngine engine_{CommEngine::COMM_ENGINE_AICPU};

    u32 ccuKernelLaunchNumStepOne_{0};
    u32 ccuKernelLaunchNumStepTwo_{0};
    u32 ccuKernelLaunchNumStepThree_{0};
    u32 ccuKernelLaunchNumStepFour_{0};

    std::vector<CcuKernelHandle> stepOneCcuKernels_;
    std::vector<CcuKernelHandle> stepTwoCcuKernels_;
    std::vector<CcuKernelHandle> stepThreeCcuKernels_;
    std::vector<CcuKernelHandle> stepFourCcuKernels_;

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;                 // 相当于之前的std::vector<InsQuePtr> tempInsQue_;
};
}

#endif