/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_BASE_V2_H
#define EXECUTOR_BASE_V2_H

#include "alg_param.h"
#include "cost_model.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_template_base.h"
#include "alg_template_register.h"
#include "utils.h"
#include "log.h"
#include "sal.h"
#include "executor_base.h"
#include "template_utils.h"
#include "order_preserved_common.h"
#include "alg_attrs.h"
#ifndef AICPU_COMPILE
#include "alg_attrs_registry.h"
#endif
#include <vector>

namespace ops_hccl {

class InsCollAlgBase {
public:
    InsCollAlgBase();
    virtual ~InsCollAlgBase();

    virtual std::string Describe() const;

    virtual std::vector<CostModelParam>
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
    {
        (void)comm;
        (void)topoInfo;
        (void)algName;
        (void)param;
        return {};
    }

    virtual AlgNetMeta GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
    {
        (void)topoInfo;
        (void)param;
        return {};
    }

    virtual HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
        = 0;

    virtual HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
    {
        (void)topoInfo;
        (void)algHierarchyInfo;
        (void)algAttrs;
        return HcclResult::HCCL_SUCCESS;
    }

    virtual HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
        = 0;

    // device
    virtual HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) = 0;

    virtual HcclResult FastLaunch(const OpParam& param, const CcuFastLaunchCtx* resCtx);

    virtual AlgAttrs GetAlgoMeta(const std::string& algName) const
    {
#ifndef AICPU_COMPILE
        const auto* attrs = AlgAttrsRegistry::Instance().Get(algName);
        return attrs != nullptr ? *attrs : AlgAttrs{};
#else
        (void)algName;
        return AlgAttrs{};
#endif
    }

    HcclResult SetTempFastLaunchAddr(
        TemplateFastLaunchCtx& tempFastLaunchCtx, void* inputPtr, void* outputPtr, const HcclMem& hcclBuff) const;

    virtual HcclResult RestoreChannelMap(
        const AlgResourceCtxSerializable& resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const;

    virtual HcclResult
    OrchestrateWithThread(const OpParam& param, const AlgResourceCtxSerializable& resCtx, ThreadHandle sendRecvThread);

#ifndef AICPU_COMPILE
    HcclResult FastLaunchSaveCtxTwoTemplate(
        const OpParam& param, const u32 threadNum, const u32 ccuKernelNum, const std::vector<ThreadHandle>& threads,
        const std::vector<u32>& ccuKernelNumList, const std::vector<std::vector<CcuKernelSubmitInfo>>& submitInfosList,
        u32 notifyNumOnMainThread) const;
#endif
protected:
    /*
     * 读取physicalLevels[levelIdx]的互联形态。Level下标与netLayer编号没有固定对应关系,
     * 一个netLayer可能贡献一级或两级, 拓扑形态必须走这里回查, 不能由下标推断。
     * 下标越界或physicalLevels为空(标准化降级)时返回COMM_TOPO_RESERVED并告警;
     * 该级无TopoInstance时字段本身就停在COMM_TOPO_RESERVED, 与前者同值, 调用方均按"形态不可用"处理。
     */
    CommTopo GetPhysicalLevelTopoType(const TopoInfoWithNetLayerDetails* topoInfo, u32 levelIdx) const;

    /*
     * 读取physicalLevels[levelIdx]上本卡各条物理链路的端口数, 降序, 按iface去重。
     * 局部量: 同一级上各rank可能不同, 不能用它做跨rank一致的决策。
     * 返回空统一表示"端口数不可用"(越界/无TopoInstance/采集降级), 不表示该级有0个端口。
     */
    std::vector<u32> GetPhysicalLevelPortNums(const TopoInfoWithNetLayerDetails* topoInfo, u32 levelIdx) const;

    inline void SetOrderPreservedBaseParams(const OrderPreservedBaseParams& params)
    {
        myRank_ = params.myRank;
        rankSize_ = params.rankSize;
        devType_ = params.devType;
        dataCount_ = params.dataCount;
        dataTypeSize_ = params.dataTypeSize;
        dataSize_ = params.dataSize;
        dataType_ = params.dataType;
        reduceOp_ = params.reduceOp;
        maxTmpMemSize_ = params.maxTmpMemSize;
    }

    // CollAlg base params
    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 rankSize_ = 0;
    HcclDevType devType_ = HcclDevType::DEV_TYPE_COUNT;

    // opInfo
    HcclReduceOp reduceOp_;
    u32 root_ = INVALID_VALUE_RANKID;
    // dataInfo
    HcclDataType dataType_;
    u64 dataCount_ = 0;

    u64 maxTmpMemSize_ = 0;

    // dataSize
    u64 dataSize_ = 0;
    u64 dataTypeSize_ = 0;
};

} // namespace ops_hccl

#endif // !HCCLV2_INS_COLL_ALG_BASE
