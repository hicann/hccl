/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_semantics_checker.h"

#include <map>
#include "check_utils.h"

namespace HcclSim {

HcclResult TaskCheckReduceSemantics(
    std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize, HcclReduceOp reduceType, RankId root,
    bool strictBufferContract)
{
    u32 rankSize = allRankMemSemantics.size();
    if (rankSize == 0) {
        return HCCL_SUCCESS;
    }

    // 对应的rank不存在需要报错
    if (allRankMemSemantics.count(root) == 0 && dataSize > 0) {
        HCCL_ERROR("Missing rank %d mem semantics", root);
        return HcclResult::HCCL_E_PARA;
    }

    u64 totalSize = 0;
    for (auto& ele : allRankMemSemantics[root][BufferType::OUTPUT]) {
        if (ele.startAddr != totalSize) {
            HCCL_ERROR(
                "[rankId:%u]Missing buffer semantic: expected startAddr is %llu, while cur buffer semantic startAddr "
                "is %llu, cur buffer semantic is %s",
                root, totalSize, ele.startAddr, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
            HCCL_ERROR(
                "[rankId:%u]cur buffer semantic reduceType %d is unequal to expected reduceType %d, cur buffer "
                "semantic is %s",
                root, ele.reduceType, reduceType, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() != rankSize) {
            HCCL_ERROR(
                "[rankId:%u]buffer semantic srcBufs size %u is unequal to rankSize %u, cur buffer semantic is %s", root,
                ele.srcBufs.size(), rankSize, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != (rankSize - 1)) {
            HCCL_ERROR(
                "[rankId:%u]cur buffer semantic srcBufs is invalid, cur buffer semantic is %s", root,
                ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        for (auto& srcBuf : ele.srcBufs) {
            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s", root,
                    ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != totalSize) {
                HCCL_ERROR(
                    "[rankId:%u]Expected semantic srcBuf srcAddr is %llu, while cur srcBuf srcAddr is %llu, cur buffer "
                    "semantic is %s",
                    root, totalSize, srcBuf.srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
        }
        totalSize += ele.size;
    }
    if (totalSize != dataSize) {
        HCCL_ERROR(
            "[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, while expected total "
            "size is %llu",
            root, totalSize, dataSize);
        return HcclResult::HCCL_E_PARA;
    }

    // [CHK_RED_002/003] 严格缓冲区契约（默认关闭，仅 AICPU 语义域用例 opt-in）：
    // - 非 root rank 的 recvBuf 不得产生任何写入语义（对称路径红线，见设计文档 §7/§8）；
    // - sendBuf 不被污染（各 rank INPUT 语义块来源必须恒为自身初始输入）。
    // 注意：非对称路径（HostDPU 等）的 API 契约从不承诺非 root recvBuf 内容，引擎将其
    // 用作暂存属合法行为，故本断言仅对显式声明的 AICPU 域启用。
    if (strictBufferContract) {
        for (auto& rankSem : allRankMemSemantics) {
            if (rankSem.first == root) {
                continue;
            }
            if (!rankSem.second[BufferType::OUTPUT].empty()) {
                const auto& firstBlock = *rankSem.second[BufferType::OUTPUT].begin();
                HCCL_ERROR(
                    "[rankId:%u]non-root output buffer should never be written, but got %zu semantic block(s), first "
                    "block is %s",
                    rankSem.first, rankSem.second[BufferType::OUTPUT].size(), firstBlock.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
        }
        for (auto& rankSem : allRankMemSemantics) {
            for (auto& ele : rankSem.second[BufferType::INPUT]) {
                if (ele.srcBufs.size() != 1) {
                    HCCL_ERROR(
                        "[rankId:%u]input buffer semantic should keep single self source, but got %zu source(s), cur "
                        "buffer semantic is %s",
                        rankSem.first, ele.srcBufs.size(), ele.Describe().c_str());
                    return HcclResult::HCCL_E_PARA;
                }
                const SrcBufDes& src = *ele.srcBufs.begin();
                if (src.rankId != rankSem.first || src.bufType != BufferType::INPUT) {
                    HCCL_ERROR(
                        "[rankId:%u]input buffer semantic source (rank %u, bufType %d) is not self INPUT, input "
                        "polluted, cur buffer semantic is %s",
                        rankSem.first, src.rankId, static_cast<int>(src.bufType), ele.Describe().c_str());
                    return HcclResult::HCCL_E_PARA;
                }
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace HcclSim
