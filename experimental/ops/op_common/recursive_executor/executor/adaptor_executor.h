/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTOR_EXECUTOR_H
#define ADAPTOR_EXECUTOR_H

#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "algo_desc.h"
#include "alg_selector.h"
#include "executor/ops_executor.h"

namespace ops_hccl {

bool IsRecursiveExecutorEnabled();

// 桥接执行器基类：继承 src 的 InsCollAlgBase，将 CalcRes/Orchestrate 转发给 recursive_executor 的 OpsExecutor。
class AdaptorExecutorBase : public InsCollAlgBase {
public:
    AdaptorExecutorBase() = default;
    ~AdaptorExecutorBase() override = default;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    std::string Describe() const override;

protected:
    std::string algName_;
    std::unique_ptr<OpsExecutor> executor_;
};

// 模板子类：编译期绑定算法名。const char* 模板参数要求变量有外部链接。
template <const char* AlgName>
class AdaptorExecutorImpl : public AdaptorExecutorBase {
public:
    AdaptorExecutorImpl() : AdaptorExecutorBase() { algName_ = AlgName; }
    ~AdaptorExecutorImpl() override = default;
};

} // namespace ops_hccl

// 用法：REGISTER_ALG(HcclCMDType::HCCL_CMD_ALLGATHER, AllGatherNHR, MakeAllGatherNhrAlgo());
#define REGISTER_ALG(cmdType, algName, algo)                                                                 \
    namespace ops_hccl {                                                                                     \
        static const char g_alg_##algName[] = #algName;                                                      \
        static bool g_reg_##algName = []() {                                                                 \
            if (!IsRecursiveExecutorEnabled()) {                                                             \
                return false;                                                                                \
            }                                                                                                \
            AlgSelector::Instance().Register(#algName, algo);                                                \
            CollAlgExecRegistryV2::Instance().Register(                                                      \
                cmdType, std::string(#algName), DefaultExecCreatorV2<AdaptorExecutorImpl<g_alg_##algName>>); \
            return true;                                                                                     \
        }();                                                                                                 \
    }

#endif // ADAPTOR_EXECUTOR_H
