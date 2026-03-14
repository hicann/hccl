/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_
#define HCCL_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_

#include "utils.h"
#include "alg_param.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

class CcuTempAllToAllVMesh1DMultiJetty : public CcuAlgTemplateBase {
public:
    explicit  CcuTempAllToAllVMesh1DMultiJetty(const OpParam& param,
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempAllToAllVMesh1DMultiJetty() override;

    std::string Describe() const override
    {
        return StringFormat("Template of AlltoAllV ccu mesh 1D MultiJetty with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;

    void SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo);

    HcclResult SetJettyNums(std::vector<uint32_t>& jettyNums, const bool multijetty) const;

private:
    A2ASendRecvInfo localSendRecvInfo_;
    CommTopo priorityTopo_ = CommTopo::COMM_TOPO_1DMESH;
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_MULTI_JETTY_H_