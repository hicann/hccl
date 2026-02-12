/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_semantics_checker.h"

#include <map>
#include "base.h"
#include "log.h"

namespace HcclSim {

HcclResult TaskCheckReduceScatterVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                            HcclReduceOp reduceType, VDataDesTag &vDataDes)
{
    u32 rankSize = allRankMemSemantics.size();
    u64 outputSize = 0;

    for (u32 rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_ERROR("Missing rank %d mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            if (ele.startAddr != totalSize) {
                HCCL_ERROR("[rankId:%u]Missing buffer semantic: exepected startAddr is %llu, "
                    "while cur buffer semantic startAddr is %llu, cur buffer semantic is %s",
                    rankId, totalSize, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                HCCL_ERROR("[rankId:%u]cur buffer semantic reduceType %d is unequal to expected reduceType %d, "
                    "cur buffer semantic is %s",
                    rankId, ele.reduceType, reduceType, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                HCCL_ERROR("[rankId:%u]buffer semantic srcBufs size %u is unequal to rankSize %u, "
                    "cur buffer semantic is %s",
                    rankId, ele.srcBufs.size(), rankSize, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != static_cast<int>(rankSize - 1)) {
                HCCL_ERROR("[rankId:%u]cur buffer semantic srcBufs is invalid, cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto &srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    HCCL_ERROR("[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, "
                        "cur buffer semantic is %s",
                        rankId, ele.Describe().c_str());
                    HcclResult::HCCL_E_PARA;
                }

                const u64 outputOffset = vDataDes.displs[rankId] * SIZE_TABLE[vDataDes.dataType];
                if (srcBuf.srcAddr != outputOffset + totalSize) {
                    HCCL_ERROR("[rankId:%u]Exepected semantic srcBuf srcAddr is %llu, "
                        "while cur srcBuf srcAddr is %llu, cur buffer semantic is %s",
                        rankId, outputOffset + totalSize, srcBuf.srcAddr, ele.Describe().c_str());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        outputSize = vDataDes.counts[rankId] * SIZE_TABLE[vDataDes.dataType];

        if (totalSize != outputSize) {
            HCCL_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
                "while expected output size is %lu", rankId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker