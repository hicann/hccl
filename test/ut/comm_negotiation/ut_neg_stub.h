/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_COMM_NEGOTIATION_STUB_H
#define UT_COMM_NEGOTIATION_STUB_H

#include <cstdint>
#include <string>
#include <vector>
#include "hccl.h"
#include "dlsym_common.h"

/*
 * comm_negotiation UT 桩配置：测试用例在 SetUp 中调用 NegStubReset() 后按需修改字段。
 * 桩对 AllReduce(MIN)/AllGather 提供"虚拟多 rank"语义：
 *   - 远端 rank 的能力位/载荷由测试注入（remoteAllReduceVals / remoteGatherPayloads）；
 *   - 本端值经 haclrtMemcpy 真实落桩内存，端到端走通 SUT 的 H2D/原语/D2H 流水。
 */
namespace ut {
// 与 comm_negotiation.cc 内 NEGOTIATION_SUCCESS_VAL/FAIL_VAL 对应的远端能力位
constexpr int32_t REMOTE_SUPPORT_VAL = 0x5a5a5a5a;
constexpr int32_t REMOTE_NOT_SUPPORT_VAL = 0x00000000;

struct NegStubConfig {
    // 通信域属性（HcclGetRankSize/HcclGetRankId/HcclGetCommName 返回值）
    uint32_t rankSize = 8;
    uint32_t myRank = 0;
    std::string commName = "ut_world";
    // 虚拟远端 rank 的 AllReduce 能力位（INT32，MIN 聚合）
    std::vector<int32_t> remoteAllReduceVals;
    // 虚拟远端 rank 的 AllGather 载荷（下标 i 对应 rank i+1）
    std::vector<std::vector<uint8_t>> remoteGatherPayloads;

    // 观测记录（用例断言用）
    HcclComm createdSubComm = nullptr; // 最近一次 HcclCreateSubCommConfig 出参
    std::string createdSubCommName;    // 子通信域名称（hcclCommName）
    int createSubCommCnt = 0;
    int allReduceCnt = 0;
    int allGatherCnt = 0;
    int commDestroyCnt = 0;
    int regCbCnt = 0;
    // 快恢注入与观测（HcclCommResume）：resumeRet注入返回值，覆盖RESUME_POST续用/失败降级两分支
    HcclResult resumeRet = HCCL_SUCCESS;
    int resumeCnt = 0;
    // EngineCtx并发销毁注入：指定tag的条目查询命中即摘除并按成功返回（模拟Probe成功入列后、
    // Release二次查询前条目已被并发路径销毁，使Release查询得NOT_FOUND），被摘除的块由桩接管统一回收
    std::string ctxEvictTagOnGet;
    // HcclEngineCtxGet调用计数（观测用）：断言性能关键路径不产生EngineCtx探测（探测未命中在
    // HCOMM侧打WARNING，逐算子探测会日志刷屏）
    int ctxGetCnt = 0;
    // 缓冲重建观测（aclrtMallocHost/aclrtFreeHost计数）：扩容=释放+重分配，同载荷复用不重建
    int mallocHostCnt = 0;
    int freeHostCnt = 0;
};

extern NegStubConfig g_negStub;

// 重置桩配置与 EngineCtx KV（每个用例 SetUp 调用）
void NegStubReset();

// 观测：comm 的 EngineCtx 中是否存在指定 tag
bool NegStubHasCtx(HcclComm comm, const std::string& tag);

// 以指定阶段触发 SUT 注册的通信域状态回调（模拟 HCOMM 生命周期事件）
HcclResult UtInvokeCommStateCb(HcclComm comm, HcclCommStatePhase phase);
} // namespace ut

#endif // UT_COMM_NEGOTIATION_STUB_H
