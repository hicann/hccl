/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_BATCH_SEND_RECV_NDA_H
#define INS_TEMP_BATCH_SEND_RECV_NDA_H

#include "alg_v2_template_base.h"

namespace ops_hccl {
// HOST(RoCE)方向复用SendWriteForNda和RecvWrite, 在接收端拷出数据后回FIN_ACK:
// 发送方暂存到本地TX半区后RoCE直写对端RX半区, 单向握手ACK/DATA_SIGNAL/FIN_ACK保护槽位复用。
// DEVICE方向保持READ语义: 接收方从对端TX半区直读用户输出，读完后通知发送方复用TX半区。
class InsTempBatchSendRecvNda : public InsAlgTemplateBase {
public:
    InsTempBatchSendRecvNda() = default;
    InsTempBatchSendRecvNda(const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempBatchSendRecvNda() override = default;

    std::string Describe() const override { return "AICPU BatchSendRecv NDA template"; }
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override {};
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {};

private:
    HcclResult RunNdaWriteTransfer(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const ChannelInfo& channel,
        ThreadHandle thread) const;
    HcclResult
    RunIntraNode(const TemplateDataParams& tempAlgParams, const ChannelInfo& channel, ThreadHandle thread) const;
};
} // namespace ops_hccl

#endif
