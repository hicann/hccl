/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 独立编译桩（参照 all_reduce_biar / recursive_executor / aicpu_task_cache 先例）。
// 被测路径（模板 CalcCostCoeff 静态方法）不执行通信原语与资源准备，本桩仅为闭合链接。
// 桩集为 cost 闭包的最小必要集（依据 CI 链接报错按目标归属逐一确证），按层分组：
// log 门控 / tuner 与环境配置 / 设备类型 / securec / HCOMM 原语 / 通道申请计算。
// 本地构建若出现残余未定义符号，按同样方式在本文件追加桩定义即可。
#include <cerrno>
#include <cstring>

#include "log.h"
#include "tuner_setup.h"
#include "alg_env_config.h"
#include "hcomm_primitives_dl.h" // 引入 HCOMM 原语声明与 ThreadHandle/ChannelHandle 等类型
#include "channel.h"             // CalcChannelRequest* 声明
#include "op_common.h"           // HcclGetHcclAlgo/CheckHostDPUOnly 声明
#include "dev_type.h"            // HcclGetDeviceType 声明（extern "C"）
#include "securec.h"             // memcpy_s 声明

// ============ log 门控 ============

bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}

bool IsErrorToWarn() { return false; }

// ============ tuner 加载态 / HCCL_ALGO 环境配置 ============

bool HcclTunerIsLoaded() { return false; }

namespace ops_hccl {
const std::map<HcclCMDType, std::vector<HcclAlgoType>> GetExternalInputHcclAlgoConfigAllType() { return {}; }

// alg_parse.cc 的 HCCL_ALGO 解析链路入口，返回空即"未配置"
std::string GetEnv(std::string idName)
{
    (void)idName;
    return "";
}

HcclResult HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo)
{
    (void)comm;
    hcclAlgo.clear();
    return HCCL_SUCCESS;
}

// executor TU 引用：非 DPU-only 拓扑
HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& hostDPUOnly)
{
    (void)comm;
    (void)topoInfo;
    hostDPUOnly = false;
    return HCCL_SUCCESS;
}

// auto_selector_base.cc NDA 判定调用，定义在 op_common.cc（本 UT 不链接），桩为不支持 NDA
HcclResult CheckSupportNda(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& isSupportNda)
{
    (void)comm;
    (void)topoInfo;
    isSupportNda = false;
    return HCCL_SUCCESS;
}

// ============ 通道申请计算（4 个被测模板的 CalcRes 路径引用，UT 不执行；声明见 channel.h） ============

HcclResult CalcChannelRequestNhr(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_E_PARA;
}

HcclResult CalcChannelRequestMesh1D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_E_PARA;
}

HcclResult CalcChannelRequestMesh1DWithPriorityTopo(
    HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    (void)priorityTopo;
    return HCCL_E_PARA;
}
} // namespace ops_hccl

// ============ 设备类型 ============

// auto_selector_base.cc 引用（声明见 dev_type.h 的 extern "C" 块）；UT 不执行，置 NOSOC
HcclResult HcclGetDeviceType(HcclDevType& devType)
{
    devType = HcclDevType::DEV_TYPE_NOSOC;
    return HCCL_SUCCESS;
}

// ============ securec（libc_sec 库符号，UT 不链接 SDK 库；先例 aicpu_task_cache/securec_stub.cc） ============

extern "C" errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    if (dest == nullptr || src == nullptr) {
        return EINVAL;
    }
    if (count > destMax) {
        if (destMax > 0) {
            std::memset(dest, 0, destMax);
        }
        return ERANGE;
    }
    std::memcpy(dest, src, count);
    return EOK;
}

// ============ HCOMM 原语（HCOMM 库符号，dlsym 运行时加载；先例 all_reduce_biar/hcomm_stub.cc） ============

extern "C" {

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

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
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

int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
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

int32_t HcommBatchModeStart(const char* batchTag)
{
    (void)batchTag;
    return 0;
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    (void)batchTag;
    return 0;
}
} // extern "C"
