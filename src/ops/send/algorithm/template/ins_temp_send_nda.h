/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SEND_NDA_H
#define INS_TEMP_SEND_NDA_H
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

// P2P NDA直写Send模板: 参考InsTempSendDpu的executor+模板分工,
// 每轮将本片数据staging到本地CCL Buffer后经Host RoCE直写对端, FIN_ACK保护CCL Buffer复用。
class InsTempSendNda : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::MESH};
    explicit InsTempSendNda();
    explicit InsTempSendNda(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempSendNda() override;

    std::string Describe() const override
    {
        std::string info = "Template of Send NDA with tempRankSize ";
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
    u32 recvRank_{0};
};

} // namespace ops_hccl
#endif // INS_TEMP_SEND_NDA_H
