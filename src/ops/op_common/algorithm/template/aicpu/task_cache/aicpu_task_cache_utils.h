/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_TASK_CACHE_UTILS_H
#define OPS_HCCL_AICPU_TASK_CACHE_UTILS_H

#include "hccl_types.h"
#include "alg_param.h"
#include <vector>

namespace ops_hccl {

class AicpuTaskCacheUtils {
public:
    static bool IsNonVariableOpType(HcclCMDType opType);
    static bool IsAlltoallvType(HcclCMDType opType);
    static HcclResult AppendVOpInfoForCache(
        const OpParam& param, uint32_t rankSize, std::vector<void*>& addrs, std::vector<uint64_t>& sizes);
    // 图模式/对称内存：[local input/output, rank0 input/output, ..., rankN-1 input/output]。
    // 无法取得完整、一致的通信域内存信息(含本地size计算失败)时返回HCCL_SUCCESS且complete=false，
    // 由调用方回退到正常展开；正式算子的参数校验仍由原执行路径负责。
    static HcclResult GetRemoteMemInfoForCache(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, std::vector<void*>& addrs,
        std::vector<uint64_t>& sizes, bool& complete);
    static HcclResult GetInputOutputInfoForCache(
        const OpParam& param, const uint32_t rankSize, uint64_t& inputSize, uint64_t& outputSize);
    // 计算 input/output 重叠特征, 供 task cache tag 拼接使用
    // diff: 重叠位置 -- output 起点相对 input 起点的字节偏移(有符号; output 在后为正, 在前为负, 相同为 0)
    // overlapBytes: 重叠宽度 -- 两区间的交叠字节数
    // 非 inplace 时两者固定为 0; 完全相同时为 (0, inputSize), 由宽度字段与普通调用(0,0)天然区分
    static void CalcInplaceDiff(
        const OpParam& param, const uint64_t inputSize, const uint64_t outputSize, int64_t& diff,
        uint64_t& overlapBytes);
};

} // namespace ops_hccl

#endif
