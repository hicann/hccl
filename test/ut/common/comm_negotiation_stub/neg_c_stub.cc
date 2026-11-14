/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * 公共协商机制共享桩实现：NegotiateCommC 强符号覆盖产品 C 弱壳；
 * IsNegotiationComm（C++本体）同签名定义（消费方UT不链接 comm_negotiation.cc）。
 * 语义见 neg_c_stub.h 注释。
 */
#include "neg_c_stub.h"
#include "comm_negotiation.h"

namespace ut {
NegCStubConfig g_negStub;

void NegCStubReset() { g_negStub = NegCStubConfig(); }
} // namespace ut

using ut::g_negStub;

namespace ops_hccl {
/* ---------------- IsNegotiationComm：C++本体桩（防嵌套门控，注入返回值） ---------------- */
bool IsNegotiationComm(HcclComm comm)
{
    (void)comm;
    return g_negStub.isNegotiationComm;
}
} // namespace ops_hccl

extern "C" {
/* ---------------- NegotiateCommC：单入口按 op 分发的数据级桩 ---------------- */
HcclResult NegotiateCommC(
    HcclComm comm, const char* typeTag, NegotiateOpC op, const void* sendBuf, uint64_t sendSize, void* recvBuf,
    uint64_t recvSize)
{
    (void)comm;
    g_negStub.lastOp = op;
    g_negStub.lastTypeTag = (typeTag == nullptr) ? std::string() : std::string(typeTag);
    switch (op) {
        case NEGOTIATE_OP_ALL_REDUCE_MIN:
            g_negStub.reduceMinCnt++;
            if (sendBuf == nullptr || recvBuf == nullptr || sendSize < sizeof(bool) || recvSize < sizeof(bool)) {
                return HCCL_E_PARA;
            }
            g_negStub.lastLocalCapability = *static_cast<const bool*>(sendBuf);
            *static_cast<bool*>(recvBuf) = g_negStub.allRanksCapable;
            return g_negStub.reduceMinRet;
        case NEGOTIATE_OP_ALL_GATHER: {
            g_negStub.gatherCnt++;
            if (g_negStub.gatherRet != HCCL_SUCCESS) {
                return g_negStub.gatherRet;
            }
            if (sendBuf == nullptr || recvBuf == nullptr || sendSize == 0 || recvSize < sendSize) {
                return HCCL_E_PARA;
            }
            // rank 数由收发尺寸推导（消费方 recvBufSize = rankSize * sendSize），与被测方零共享配置
            const uint64_t rankNums = recvSize / sendSize;
            const auto* src = static_cast<const uint8_t*>(sendBuf);
            for (uint64_t rank = 0; rank < rankNums; rank++) {
                auto* slot = static_cast<uint8_t*>(recvBuf) + rank * sendSize;
                // 保真 AllGather 语义：每个 rank 槽位先复制本端载荷（全域同源基线）
                for (uint64_t i = 0; i < sendSize; i++) {
                    slot[i] = src[i];
                }
                // 再应用槽位内字段覆盖（驱动"存在低配 rank"/"静默假成功"等差异化用例）
                for (const auto& item : g_negStub.gatherOverrides) {
                    const uint64_t offset = item.first;
                    const std::vector<uint8_t>& bytes = item.second;
                    for (size_t j = 0; j < bytes.size() && offset + j < sendSize; j++) {
                        slot[offset + j] = bytes[j];
                    }
                }
            }
            return HCCL_SUCCESS;
        }
        default:
            return HCCL_E_PARA;
    }
}
} // extern "C"
