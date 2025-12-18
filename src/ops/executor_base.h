/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_BASE_H
#define EXECUTOR_BASE_H

#include "alg_param.h"
#include "topo.h"
#include "channel.h"
#include "alg_template_base.h"
#include "alg_template_register.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"

namespace ops_hccl {

// 节点间RDMA发送数据单个WQE支持的最大数据量
const u64 RDMA_SEND_MAX_SIZE = 0x80000000;
// 节点内单个SDMA任务发送数据支持的最大数据量
const u64 SDMA_SEND_MAX_SIZE = 0x100000000;
constexpr u32 HCCL_INTERNODE_MAX_DATA_RATE = 1; // node间通信的单次通信量最多为node通信量的1倍（R-HD或NHR）
constexpr s32 LEVEL0_PLANE_NUM_IN_NPRING_SINGLE = 1;
constexpr s32 LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE = 2;

struct ExecMem {
    u64 count = 0;
    HcclMem inputMem;           /* 单算子模式时是InCCLMem, 图模式时是InUserMem */
    HcclMem outputMem;          /* 单算子模式时是OutCCLMem, 图模式时是OutUserMem */
    HcclMem scratchMem;
    void *inputPtr = nullptr;   /* InUserMem的地址，图模式时与inputMem的地址相同 */
    void *outputPtr = nullptr;  /* OutUserMem的地址，图模式时与outputMem的地址相同 */
};

class ExecutorBase {
public:
    ExecutorBase();
    virtual ~ExecutorBase() = default;
    // 资源计算接口
    virtual HcclResult CalcResRequest(HcclComm comm, const OpParam& param,
        TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
        AlgResourceRequest& resourceRequest, AlgType& algType) = 0;
    // 算法编排接口
    virtual HcclResult Orchestrate(const OpParam &param, AlgResourceCtx* resCtx) = 0;

    virtual HcclResult KernelRun(const OpParam &param, ExecMem &execMem) = 0;
    inline AlgDesc GetAlgDesc() {return desc_;}
protected:
    HcclResult GetSubCommInfo(const CommPlane levelIndex, SubCommInfo &info);
    HcclResult RefreshAlgType(AlgType& algType);
    std::string tag_;
    HcclWorkflowMode workflowMode_;
    TopoInfo* topoInfo_ = nullptr;
    AlgResourceCtx* algResource_ = nullptr;
    AlgType algType_;
    u32 level0RankSize = 0;
    AlgDesc desc_;
};
}
#endif