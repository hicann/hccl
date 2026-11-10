/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_RECV_NDA_H
#define INS_TEMP_RECV_NDA_H
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include "alg_v2_template_base.h"
#include "alg_v2_template_register.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "hccl_res.h"

namespace ops_hccl {

// P2P NDA直写Recv模板: 参考InsTempRecvDpu的executor+模板分工,
// 每轮等对端直写本端CCL Buffer后搬运到output, FIN_ACK+同步保护CCL Buffer复用。
class InsTempRecvNda : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::MESH};
    explicit InsTempRecvNda();
    explicit InsTempRecvNda(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempRecvNda() override;

    std::string Describe() const override
    {
        std::string info = "Template of Recv NDA with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override {};
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {};

private:
    u32 sendRank_{0};
};

} // namespace ops_hccl
#endif // INS_TEMP_RECV_NDA_H
