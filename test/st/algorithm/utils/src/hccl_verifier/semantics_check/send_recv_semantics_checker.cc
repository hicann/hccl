/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_recv_semantics_checker.h"

#include <map>
#include "hccl/base.h"
#include "check_utils.h"

namespace HcclSim {
    HcclResult TaskCheckSendRecvSemantics(
        std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize, RankId srcRank, RankId dstRank) {
        u32 rankSize = allRankMemSemantics.size();

        // 对应的rank不存在需要报错
        if (!allRankMemSemantics.count(dstRank)) {
            HCCL_ERROR("[rankId:%u]Missing buffer semantic.", dstRank);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (auto &ele: allRankMemSemantics[dstRank][BufferType::OUTPUT]) {
            if (ele.startAddr != totalSize) {
                HCCL_ERROR(
                    "[rankId:%u]Missing buffer semantic: expected startAddr is %llu, while cur buffer semantic "
                    "startAddr is %llu, cur buffer semantic is %s",
                    dstRank, totalSize, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic should not be reduce, which mean srcBufs size should be 1, "
                    "while cur buffer semantic is %s",
                    dstRank, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->rankId != srcRank) {
                HCCL_ERROR(
                    "[rankId:%u]Buffer semantic srcBuf rank[%u] is not from srcRank[%u], cur buffer semantic is %s",
                    dstRank, ele.srcBufs.begin()->rankId, srcRank, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s",
                    dstRank, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            if (ele.srcBufs.begin()->srcAddr != totalSize) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, while it is %llu, cur buffer "
                    "semantic is %s",
                    dstRank, totalSize, ele.srcBufs.begin()->srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize) {
            HCCL_ERROR(
                "[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, which should be %llu",
                dstRank, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }

        return HcclResult::HCCL_SUCCESS;
    }
} // namespace HcclSim
