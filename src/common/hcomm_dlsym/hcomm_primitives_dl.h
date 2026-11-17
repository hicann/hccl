/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PRIMITIVES_DL_H
#define HCOMM_PRIMITIVES_DL_H

#include "dlsym_common.h"
#include "hcomm_primitives.h" // 原头文件，包含所有类型和定义
#include "hccl_rank_graph.h"  // EndpointDesc（与HCOMM侧hcomm_res_defs.h同布局的公共结构体）
#include "hccl_types.h"

/* 8.5.0 桩: HcclCommSymWindow (来自 hccl_types.h) */
#if CANN_VERSION_NUM < CANN_VERSION(9, 0, 0)
typedef void* HcclCommSymWindow;
#endif

/* HcommBatchTransferOnThread 及其描述符由 HCOMM 仓提供。HCCL 使用私有 ABI 兼容类型，
 * 避免在旧 HCOMM 头文件未声明新类型时编译失败。
 */
typedef enum {
    HCCL_HCOMM_TRANSFER_TYPE_INVALID = -1,
    HCCL_HCOMM_TRANSFER_TYPE_WRITE = 0,
    HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE = 1,
    HCCL_HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,
    HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3,
    HCCL_HCOMM_TRANSFER_TYPE_READ = 4,
    HCCL_HCOMM_TRANSFER_TYPE_READ_REDUCE = 5,
    HCCL_HCOMM_TRANSFER_TYPE_NOTIFY_RECORD = 6,
    HCCL_HCOMM_TRANSFER_TYPE_NOTIFY_WAIT = 7,
    HCCL_HCOMM_TRANSFER_TYPE_NOTIFY_WAIT_WITH_DEFAULT_TIMEOUT = 8
} HcclHcommTransferType;

typedef struct {
    HcclHcommTransferType transType;
    uint8_t reserved[4];
    union {
        uint8_t raws[56];
        struct {
            uint64_t len;
            void* dst;
            void* src;
        } write;
        struct {
            uint64_t len;
            void* dst;
            void* src;
        } read;
        struct {
            uint64_t count;
            void* dst;
            void* src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
        } reduce;
        struct {
            uint32_t notifyIdx;
        } notifyRecord;
        struct {
            uint64_t len;
            void* dst;
            void* src;
            uint32_t notifyIdx;
        } writeWithNotify;
        struct {
            uint64_t count;
            void* dst;
            void* src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
            uint32_t notifyIdx;
        } writeReduceWithNotify;
    } transferInfo;
} HcclHcommBatchTransferDesc;

/* HcommEndpointCheckFeature 的 featureType 取值。
 * 与 HCOMM 仓 include/hcomm_res_defs.h 的 HcommEndpointFeatureType 逐值锁定对齐（跨仓协议，勿单侧改动）；
 * HCOMM 侧该枚举为 C enum（ABI 上 4 字节 int），此处以同值枚举独立声明，经 dlsym 符号跨仓传递。
 * 注: HCOMM 侧对应成员名为 HCOMM_ENDPOINT_FEATURE_INLINE_REDUCE（无DPU_前缀），成员名可不同, 数值必须一致 */
enum HcclHcommFeatureId {
    HCCL_HCOMM_FEATURE_INVALID = -1,
    HCCL_HCOMM_FEATURE_NDA = 0,               /* NPU Direct RDMA Async 特性 */
    HCCL_HCOMM_FEATURE_DPU_INLINE_REDUCE = 1, /* 本端 hostDPU 是否支持 inline reduce（传输带规约语义） */
};

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(int32_t, HcommThreadSynchronize, ThreadHandle thread);
DECL_WEAK_FUNC(
    int32_t, HcommSendRequest, uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId);
DECL_WEAK_FUNC(int32_t, HcommWaitResponse, uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId);
DECL_WEAK_FUNC(HcclResult, HcommThreadJoin, ThreadHandle thread, uint32_t timeout);
#ifdef HCOMM_TIMEOUT_FLOAT_TYPE
DECL_WEAK_FUNC(int32_t, HcommThreadResAcquireTimeOut, float timeOut);
DECL_WEAK_FUNC(int32_t, HcommSetNotifyWaitTimeOut, float timeOut);
#else
DECL_WEAK_FUNC(int32_t, HcommThreadResAcquireTimeOut, uint32_t timeOut);
DECL_WEAK_FUNC(int32_t, HcommSetNotifyWaitTimeOut, uint32_t timeOut);
#endif
DECL_WEAK_FUNC(int32_t, HcommThreadNotifyWaitOnThreadWithDefaultTimeout, ThreadHandle thread, uint32_t notifyIdx);
DECL_WEAK_FUNC(
    int32_t, HcommChannelNotifyWaitOnThreadWithDefaultTimeout, ThreadHandle thread, ChannelHandle channel,
    uint32_t localNotifyIdx);
DECL_WEAK_FUNC(int32_t, HcommChannelNotifyWaitWithDefaultTimeout, ChannelHandle channel, uint32_t localNotifyIdx);
DECL_WEAK_FUNC(
    int32_t, HcommWriteWithNotifyNbiOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t len, uint32_t remoteNotifyIdx);
DECL_SUPPORT_FLAG(HcommWriteReduceWithNotifyOnThread);
DECL_WEAK_FUNC(int32_t, HcommChannelFenceOnThread, ThreadHandle thread, ChannelHandle channel);
DECL_WEAK_FUNC(int32_t, HcommChannelDrainOnThread, ThreadHandle thread, ChannelHandle channel);
DECL_SUPPORT_FLAG(HcommBatchTransferOnThread);
DECL_SUPPORT_FLAG(HcommThreadResAcquireTimeOut);
DECL_SUPPORT_FLAG(HcommSetNotifyWaitTimeOut);
DECL_SUPPORT_FLAG(HcommThreadNotifyWaitOnThreadWithDefaultTimeout);
DECL_SUPPORT_FLAG(HcommChannelNotifyWaitOnThreadWithDefaultTimeout);
DECL_SUPPORT_FLAG(HcommChannelNotifyWaitWithDefaultTimeout);
int32_t HcclHcommBatchTransferOnThread(
    ThreadHandle thread, ChannelHandle channel, const HcclHcommBatchTransferDesc* transferDescs,
    uint32_t transferDescNum);
bool IsHcommDefaultTimeoutSupported();
HcclResult HcclSetNotifyWaitTimeOut(uint32_t timeout);
HcclResult HcclThreadResAcquireTimeOut(uint32_t timeout);
HcclResult HcclThreadNotifyWaitOnThreadDefault(ThreadHandle thread, uint32_t notifyIdx, uint32_t fallbackTimeout);
HcclResult HcclChannelNotifyWaitOnThreadDefault(
    ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t fallbackTimeout);
HcclResult HcclChannelNotifyWaitDefault(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t fallbackTimeout);

DECL_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheLookup, const char* tag, bool* isHit);
DECL_SUPPORT_FLAG(HcommAicpuTsTaskCacheLookup);
DECL_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheStart, const char* tag, void** addrs, uint64_t* sizes, uint64_t count);
DECL_SUPPORT_FLAG(HcommAicpuTsTaskCacheStart);
DECL_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheEnd, const char* tag);
DECL_SUPPORT_FLAG(HcommAicpuTsTaskCacheEnd);
DECL_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheExecute, const char* tag, void** addrs, uint64_t* sizes, uint64_t count);
DECL_SUPPORT_FLAG(HcommAicpuTsTaskCacheExecute);
DECL_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheClear, const char* tag);
DECL_SUPPORT_FLAG(HcommAicpuTsTaskCacheClear);

/* 9.1.0 新增 Endpoint 特性查询接口（如 NDA），经 dlsym 弱符号绑定；
 * HcommEndpointFeatureType/EndpointDesc 类型由 dlsym_common.h 已包含的 hcomm_res_defs.h 提供 */
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
DECL_WEAK_FUNC(
    HcommResult, HcommEndpointCheckFeature, HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc,
    bool* value);
DECL_SUPPORT_FLAG(HcommEndpointCheckFeature);

/* DPU inline reduce 复合支持判定：能力查询符号 + write-reduce 传输符号，任一缺失按不支持处理
 * （HcommIsSupportCcuV2 复合判定先例） */
bool IsHcommDpuInlineReduceSupported();
/* 本端 hostDPU inline reduce 能力查询（本端语义，无跨rank交换）：
 * 按HCOMM真实签名以本端host侧EndpointDesc查询；符号缺失/查询失败一律按不支持降级、禁止上抛；
 * 全域聚合由 HCCL 公共协商机制完成 */
HcclResult HcclCheckDpuInlineReduceLocal(const EndpointDesc& hostDpuEndpoint, bool* supported);
#endif

void HcommPrimitivesDlInit(void* libHcommHandle); // 本模块独立初始化

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PRIMITIVES_DL_H
