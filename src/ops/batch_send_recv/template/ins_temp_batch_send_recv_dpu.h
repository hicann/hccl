/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_BATCH_SEND_RECV_DPU
#define INS_TEMP_BATCH_SEND_RECV_DPU
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

class InsTempBatchSendRecvDpu : public InsAlgTemplateBase {
public:
    explicit InsTempBatchSendRecvDpu();
    explicit InsTempBatchSendRecvDpu(const OpParam &param, const u32 rankId,  // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempBatchSendRecvDpu() override;

    std::string Describe() const override
    {
        std::string info = "Template of BatchSendRecv with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
        TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest) override;
    HcclResult DPUKernelRun(const TemplateDataParams &tempAlgParam,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const u32 myRank,
        const std::vector<std::vector<uint32_t>> &subCommRanks);
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override{};
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override{};

private:
    u32 recvRank_{0};
    u64 count_{0};
    u64 processSize_{0};
    u64 dataCount_{0};
    u64 dataTypeSize_{0};
    HcclDataType dataType_;
    u64 dataSize_{0};
    u64 hcclbuffBlockMemSize_{0};
    ChannelInfo sendRecvChannel_;  // 只有一个channel
    ChannelInfo subSendRecvChannel_; // aicpu对应第二条channel
    ThreadHandle thread_;      // 只涉及一个thread
    ThreadHandle subThread_;      // aicpu可以用2 thread？
};

}  // namespace ops_hccl
// #endif  //OPEN_HCCL_INS_TEMP_BATCH_SEND_RECV_DPU
#endif