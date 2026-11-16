/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 独立编译桩：仅闭合链接，不执行通信与日志（先例 common/template_utils/ut_stub.cc）。
// 注册/选路 UT 链接 executor 注册 TU 的真闭包（真注册表 + 真过滤缝隙），本文件桩掉闭包
// 引用的运行期外部依赖（HCOMM 原语 / tuner / 环境变量），选路断言不触达这些路径。
// 若本地构建出现残余未定义符号，按同样方式在本文件追加桩定义即可。
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "log.h"
#include "hccl_sym_win_dl.h"
#include "hcomm_primitives_dl.h"
#include "alg_parse.h"
#include "sal.h"
#include "topo.h"

// ============ log 门控 ============

bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}

bool IsErrorToWarn() { return false; }

// ============ HCOMM 对称窗口（本 UT 不触达该路径） ============

HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    (void)winHandle;
    (void)offset;
    (void)peerRank;
    (void)ptr;
    return HcclResult::HCCL_E_INTERNAL;
}

// ============ RankGraph 拓扑查询（CANN hccl_rank_graph.h 强声明；UT 不触达建链路径） ============

extern "C" HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    (void)comm;
    if (netLayers != nullptr) {
        *netLayers = nullptr;
    }
    if (netLayerNum != nullptr) {
        *netLayerNum = 0;
    }
    return HcclResult::HCCL_E_INTERNAL;
}

extern "C" HcclResult HcclRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    (void)comm;
    (void)netLayer;
    (void)srcRank;
    (void)dstRank;
    if (links != nullptr) {
        *links = nullptr;
    }
    if (linkNum != nullptr) {
        *linkNum = 0;
    }
    return HcclResult::HCCL_E_INTERNAL;
}

// ============ ops_hccl 运行期依赖桩（真闭包引用、选路断言不触达） ============

namespace ops_hccl {
u64 GetDebugConfig() { return 0; }

std::string GetEnv(std::string name)
{
    (void)name;
    return "";
}

HcclResult HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo)
{
    (void)comm;
    hcclAlgo = "";
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& hostDPUOnly)
{
    (void)comm;
    (void)topoInfo;
    hostDPUOnly = false;
    return HcclResult::HCCL_SUCCESS;
}

// auto_selector_base.cc NDA 判定调用，定义在 op_common.cc（本 UT 不链接），桩为不支持 NDA（先例
// reduce_scatter_seq64/ut_stub.cc）
HcclResult CheckSupportNda(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& isSupportNda)
{
    (void)comm;
    (void)topoInfo;
    isSupportNda = false;
    return HcclResult::HCCL_SUCCESS;
}

const std::map<HcclCMDType, std::vector<HcclAlgoType>> GetExternalInputHcclAlgoConfigAllType() { return {}; }

const u32& GetExternalInputIntraRoceSwitch()
{
    static u32 val = 0;
    return val;
}

const bool& GetExternalInputInterHccsDisable()
{
    static bool val = false;
    return val;
}

HcclResult GetUserRankBySubCommRank(u32 subCommRank, u32 curLevel, AlgHierarchyInfo& algHierarchyInfo, u32& userRank)
{
    (void)subCommRank;
    (void)curLevel;
    (void)algHierarchyInfo;
    userRank = 0;
    return HcclResult::HCCL_E_PARA;
}

HcclResult GetUbMultiChannelNum(HcclComm comm, u32& multiChannelNum)
{
    (void)comm;
    multiChannelNum = 1;
    return HcclResult::HCCL_E_NOT_SUPPORT;
}
} // namespace ops_hccl

// ============ tuner（恒未加载） ============

bool HcclTunerIsLoaded() { return false; }

HcclResult HcclTunerInit(HcclComm comm, const ops_hccl::TopoInfoWithNetLayerDetails* topoInfo)
{
    (void)comm;
    (void)topoInfo;
    return HcclResult::HCCL_SUCCESS;
}

// ============ 外部 C 接口（设备类型 / batch 模式切换 / 批量传输原语） ============

extern "C" {
HcclResult HcclGetDeviceType(HcclDevType& devType)
{
    devType = HcclDevType::DEV_TYPE_910B;
    return HcclResult::HCCL_SUCCESS;
}

// dlsym 版本查询桩：低版本使 Drain compat 走 Fence 回退路径（UT 不执行数据面）
int GetHcommVersion(void) { return 0; }

// 失败回退桩（先例 selector_engine/securec 桩）：UT 不触达 engine ctx 创建路径
HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    if (ctx == nullptr) {
        return HcclResult::HCCL_E_PARA;
    }
    *ctx = malloc(size > 0 ? static_cast<size_t>(size) : 1);
    return (*ctx != nullptr) ? HcclResult::HCCL_SUCCESS : HcclResult::HCCL_E_PARA;
}

int32_t HcommBatchModeEnd(const char* tag)
{
    (void)tag;
    return 0;
}

int32_t HcommBatchModeStart(const char* tag)
{
    (void)tag;
    return 0;
}

// ============ KernelRun host 流程（eager 舞步）桩：PostLocalCopy 契约 UT ============
// UT 不经真实 DPU 请求：请求/响应配对固定 msgId（recvMsgId==sendMsgId 才能走到 PostLocalCopy）
int32_t HcommThreadSynchronize(ThreadHandle thread)
{
    (void)thread;
    return 0;
}

int32_t HcommSendRequest(uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId)
{
    (void)handle;
    (void)msgTag;
    (void)src;
    (void)sizeByte;
    if (msgId != nullptr) {
        *msgId = 42u;
    }
    return 0;
}

int32_t HcommWaitResponse(uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId)
{
    (void)handle;
    (void)dst;
    (void)sizeByte;
    if (msgId != nullptr) {
        *msgId = 42u;
    }
    return 0;
}

// 批量传输原语（hcomm_dlsym 强声明路径）：UT 不执行数据面，桩值不影响选路断言
int32_t HcclHcommBatchTransferOnThread(
    ThreadHandle thread, ChannelHandle channel, const HcclHcommBatchTransferDesc* transferDescs,
    uint32_t transferDescNum)
{
    (void)thread;
    (void)channel;
    (void)transferDescs;
    (void)transferDescNum;
    return 0;
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    (void)thread;
    (void)channel;
    (void)localNotifyIdx;
    return 0;
}

// 通道级等待（带超时；hcomm_primitives_dl 弱声明路径的强引用回退桩）
int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t notifyIdx, uint32_t timeout)
{
    (void)thread;
    (void)channel;
    (void)notifyIdx;
    (void)timeout;
    return 0;
}

// DPU 数据面 Fence（CANN 强声明路径；UT 不执行数据面）
int32_t HcommFenceOnThread(ThreadHandle thread)
{
    (void)thread;
    return 0;
}

// HCOMM 数据面原语家族（CANN hcomm_primitives.h 强声明；模板/包装层编译进闭包但 UT 不执行）
// LocalCopy 实参录制口（wrapper 层已合成 addr+offset）：供 PostLocalCopy 契约 UT 反推
// DataSlice 偏移。单线程 UT 无需加锁。定义与测试文件中的声明保持一致（ODR）。
struct UtLocalCopyCall {
    const void* src;
    void* dst;
    uint64_t len;
};
std::vector<UtLocalCopyCall> g_utLocalCopyCalls;

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    g_utLocalCopyCalls.push_back(UtLocalCopyCall{src, dst, len});
    return 0;
}

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return 0;
}

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    (void)thread;
    (void)dstThread;
    (void)dstNotifyIdx;
    return 0;
}

// master HostDpu 新方案重构（4dbf9150）在 NHR DPU inter-node 的 isDpuThreadEnabled 分支引入
// 的线程 notify 原语；UT 闭包链接该模板但选路断言不触达数据面，桩为立即成功
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    (void)thread;
    (void)notifyIdx;
    (void)timeOut;
    return 0;
}

int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return 0;
}

int32_t HcommWriteReduceWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    (void)remoteNotifyIdx;
    return 0;
}

int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

int32_t HcommReadReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return 0;
}

// 带默认超时的等待（hcomm_dlsym 强声明路径）
HcclResult HcclThreadNotifyWaitOnThreadDefault(ThreadHandle thread, uint32_t notifyIdx, uint32_t fallbackTimeout)
{
    (void)thread;
    (void)notifyIdx;
    (void)fallbackTimeout;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclChannelNotifyWaitOnThreadDefault(
    ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t fallbackTimeout)
{
    (void)thread;
    (void)channel;
    (void)localNotifyIdx;
    (void)fallbackTimeout;
    return HcclResult::HCCL_SUCCESS;
}
} // extern "C"

// ============ securec 桩（UT 环境不链接 libc_sec，先例 selector_engine/securec_stub.cc） ============

extern "C" errno_t memcpy_s(void* dest, size_t numberOfElements, const void* src, size_t count)
{
    if (dest == nullptr || src == nullptr) {
        return EINVAL;
    }
    if (numberOfElements < count) {
        return ERANGE;
    }
    (void)memcpy(dest, src, count);
    return EOK;
}

extern "C" int sprintf_s(char* strDest, size_t destMax, const char* format, ...)
{
    if (strDest == nullptr || format == nullptr || destMax == 0) {
        return -1;
    }
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(strDest, destMax, format, args);
    va_end(args);
    if (ret < 0 || static_cast<size_t>(ret) >= destMax) {
        return -1;
    }
    return ret;
}

extern "C" errno_t memset_s(void* dest, size_t numberOfElements, int c, size_t count)
{
    if (dest == nullptr) {
        return EINVAL;
    }
    if (numberOfElements < count) {
        return ERANGE;
    }
    (void)memset(dest, c, count);
    return EOK;
}

extern "C" errno_t strcpy_s(char* strDest, size_t numberOfElements, const char* strSrc)
{
    if (strDest == nullptr || strSrc == nullptr) {
        return EINVAL;
    }
    size_t srcLen = strlen(strSrc);
    if (numberOfElements <= srcLen) {
        return ERANGE;
    }
    (void)memcpy(strDest, strSrc, srcLen + 1);
    return EOK;
}
