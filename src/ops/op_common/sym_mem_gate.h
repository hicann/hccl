/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SYM_MEM_GATE_H
#define OPS_HCCL_SYM_MEM_GATE_H

#include <string>
#include "inc/alg_param.h"

namespace ops_hccl {

// AllReduce/ReduceScatter/Reduce共用的对称内存限制：64位数据类型与PROD规约不支持，
// 与算子入口Selector后的限制条件保持一致。
bool IsReduceOpSymMemAllowed(const OpParam& param);

// AllReduce/ReduceScatter/Reduce 共用的对称内存确定性门禁。
// AllReduce/ReduceScatter：SoleNHR 为单模板单棵全 rank 蝶形树，多级仅改变链路层、规约顺序仍固定，
// 不受 topoLevelNums 限制，故先于拓扑级数检查放行；其余多级算法 L0/L1 两棵树参与规约，确定性未验证；
// 单级白名单算法均由单一模板完成规约、规约写者按步序固定，确定性成立。
// Reduce：仅放行多级 root 侧零拷贝白名单。
bool IsReduceOpSymMemDeterministic(
    const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo, HcclCMDType opType);

// 仅 Scatter 与 Reduce 跳过 input 侧探测：Scatter 非 root sendBuf 可空/虚值；Reduce 只写 root 的 output 窗口，
// 不应额外要求注册 sendBuf。其余算子（含 RSV）保持双侧探测。
bool NeedSymMemInputSide(HcclCMDType opType);

// 与CheckAndSetSymmetricMemory的探测范围一致：每一侧需要探测的buffer都拿到了对称窗口才算有效
bool IsSymMemWindowValid(const OpParam& param);

} // namespace ops_hccl

#endif // OPS_HCCL_SYM_MEM_GATE_H
