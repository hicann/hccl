/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOSTDPU_INLINE_REDUCE_H
#define HOSTDPU_INLINE_REDUCE_H

#include "hccl.h"
#include "alg_param.h"

namespace ops_hccl {
/* hostDPU inline reduce 能力协商功能（消费公共协商机制 comm_negotiation，机制与策略分离）：
 * 本模块拥有「协商什么」（门控/本端能力查询/全域聚合/三态终态缓存），协商机制由 comm_negotiation 提供。
 * UT（test/ut/hostdpu_inline_reduce）对 comm_negotiation 接口打桩，与公共协商模块UT（test/ut/comm_negotiation）解耦。
 */

// hostDPU inline reduce 独立协商阶段（per-comm一次性）：topoInfo计算后、路由前执行，
// 全域聚合本端能力位，结果写入per-comm终态缓存并设置topoInfo->allDpuSupportInlineReduce，
// 供后续cost model构建期topoCustomCheck（新selector路径）与legacy SelectDPUAlgo（显式分支）读取。
// 门控（五条件任一命中即跳过）：
// 1. P2P算子（SEND/RECEIVE/BATCH_SEND_RECV）：P2P语义仅要求配对rank调用，其余rank可不进入Selector，
//    全域协商AllReduce会因对端永不到达而挂死；且P2P路径不消费allDpuSupportInlineReduce
// 2. 非hostDPU组网；3. 非A5(950)及后续代际；4. 协商子通信域（防递归主防线，协商AllReduce会重入Selector）
// 5. 图模式捕获态（opStream处于capture）：录制期执行全域集合通信+同步会把"录制可各rank异步推进"
//    变成全域同步点，且失败会写NEGOTIATION_DEGRADED终态缓存不可自愈；跳过时不写缓存
//    （缓存命中读取不受影响），能力位保持默认false，replay后的非捕获态op缓存miss时正常协商
// 注意：不得以env/per-op属性门控本协商——env是per-process的，异构env会使部分rank进入协商集合
// 而其余rank跳过，导致协商集合通信挂死；env/per-op门控属于使能层（opCustomCheck/legacy分支）
HcclResult NegotiateHostDpuInlineReduceCapability(
    HcclComm comm, HcclCMDType opType, aclrtStream opStream, TopoInfoWithNetLayerDetails* topoInfo);
} // namespace ops_hccl

#endif // HOSTDPU_INLINE_REDUCE_H
