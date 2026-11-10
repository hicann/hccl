/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <set>
#include <cstring>
#include <memory>
#include "securec.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "hccl/hccl_types.h"
#include "sim_world.h"
#include "sim_stream.h"
#include "hccl_sim_pub.h"
#include "log.h"
#include "alg_param.h"
#include "sim_task_queue.h"
#include "rt_external.h"

using namespace hccl;
using namespace ops_hccl;
thread_local uint32_t curr_dev_id = UINT32_MAX;

namespace {
// 登记 aclrtMalloc else 分支用宿主 malloc 模拟的"设备"指针, 使 aclrtFree 能真实释放, 避免仿真长跑泄漏
// (ST 各 rank 在独立 std::thread 中并发调用, 需加锁)
std::mutex g_stubMallocMutex;
std::set<void*> g_stubMallocPtrs;
char g_aicpuKernelHandle;

// SimWorld无真实context对象，句柄直接按deviceId编码（+1避开device 0编码为nullptr），使context桩
// 对齐真实ACL语义：context携带device绑定，跨线程传递句柄即传递device归属。CommWorker将发起线程
// ctx下发到worker线程，经SetCurrentContext恢复device绑定；否则worker线程thread_local curr_dev_id
// 保持UINT32_MAX，task链路触及依赖它的桩（aclrtMalloc/aclrtCreateStream等按rankId定位SimNpu）会异常
aclrtContext DeviceIdToCtx(uint32_t devId) { return reinterpret_cast<aclrtContext>(static_cast<uintptr_t>(devId + 1)); }

uint32_t CtxToDeviceId(aclrtContext ctx) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ctx)) - 1; }
} // namespace

extern "C" unsigned int HcclLaunchAicpuKernel(OpParam* param);

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
aclError aclrtFreeHost(void* hostPtr)
{
    if (hostPtr != nullptr) {
        free(hostPtr);
        hostPtr = nullptr;
    }
    return ACL_SUCCESS;
}

aclError aclrtMallocHost(void** hostPtr, size_t size)
{
    if (hostPtr == nullptr || size == 0) {
        HCCL_ERROR("[aclrtMallocHost] invalid input hostPtr or size");
        return ACL_ERROR_INVALID_PARAM;
    }

    *hostPtr = malloc(size);
    if (*hostPtr == nullptr) {
        HCCL_ERROR("[aclrtMallocHost] malloc host memory failed, size[%zu]", size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    return ACL_SUCCESS;
}

// 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    if (devPtr == nullptr || size == 0) {
        HCCL_ERROR("[aclrtMalloc] invalid input devPtr or size");
        return ACL_ERROR_INVALID_PARAM;
    }
    u32 memType = static_cast<u32>(policy);
    HcclSim::SimNpu& simNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    if (memType == BUFFER_INPUT_MARK) {
        *devPtr = reinterpret_cast<void*>(simNpu.AllocMemory(BufferType::INPUT, size));
    } else if (memType == BUFFER_OUTPUT_MARK) {
        *devPtr = reinterpret_cast<void*>(simNpu.AllocMemory(BufferType::OUTPUT, size));
    } else {
        *devPtr = malloc(size);
        if (*devPtr == nullptr) {
            HCCL_ERROR("[aclrtMalloc] malloc failed, size[%zu]", size);
            return ACL_ERROR_INTERNAL_ERROR;
        }
        std::lock_guard<std::mutex> lock(g_stubMallocMutex);
        g_stubMallocPtrs.insert(*devPtr);
    }
    return ACL_SUCCESS;
}

aclError aclrtFree(void* devPtr)
{
    {
        std::lock_guard<std::mutex> lock(g_stubMallocMutex);
        if (g_stubMallocPtrs.erase(devPtr) > 0) {
            free(devPtr);
            return ACL_SUCCESS;
        }
    }
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId_)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetThreadLastTaskId(uint32_t* taskId)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    if (dst == nullptr || src == nullptr) {
        HCCL_ERROR("[aclrtMemcpy] invalid input dst or src");
        return ACL_ERROR_INVALID_PARAM;
    }

    if (count == 0 || destMax < count) {
        HCCL_ERROR("[aclrtMemcpy] invalid input count[%llu] or destMax[%llu]", count, destMax);
        return ACL_ERROR_INVALID_PARAM;
    }

    memcpy(dst, src, count);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count)
{
    if (devPtr == nullptr || count > maxCount) {
        HCCL_ERROR("[aclrtMemset] invalid input param.");
        return ACL_ERROR_INVALID_PARAM;
    }

    memset(devPtr, value, count);
    return ACL_SUCCESS;
}

const std::map<HcclDevType, std::string> DEV_VERSION_MAP
    = {{HcclDevType::DEV_TYPE_910B, "Ascend910B1"}, {HcclDevType::DEV_TYPE_910_93, "Ascend910_9391"}};

const char* aclrtGetSocName()
{
    auto npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    auto devType = npu.GetDevType();
    auto it = DEV_VERSION_MAP.find(devType);
    if (it != DEV_VERSION_MAP.end()) {
        HCCL_INFO("[aclrtGetSocName] devType[%u], socVersion[%s]", devType, it->second.c_str());
        return it->second.c_str();
    }

    HCCL_ERROR("[aclrtGetSocName] can not find devType[%u] in DEV_VERSION_MAP", devType);
    return "";
}

HcclResult HcclGetDeviceType(HcclDevType& devType)
{
    auto npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    devType = npu.GetDevType();
    return HCCL_SUCCESS;
}

aclError aclrtGetDevice(int32_t* device)
{
    *device = curr_dev_id;
    return ACL_SUCCESS;
}

// 已绑定device的线程返回其device对应的非空context句柄；未绑定（curr_dev_id为UINT32_MAX）返回null，
// 调用方（如CommWorker::WorkerLoop）走"ctx为null则跳过SetCurrent"降级路径
aclError aclrtGetCurrentContext(aclrtContext* context)
{
    if (context == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *context = (curr_dev_id == UINT32_MAX) ? nullptr : DeviceIdToCtx(curr_dev_id);
    return ACL_SUCCESS;
}

// 按句柄恢复当前线程device绑定（对齐真实ACL：SetCurrentContext同时切换线程当前device）
aclError aclrtSetCurrentContext(aclrtContext context)
{
    if (context == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    curr_dev_id = CtxToDeviceId(context);
    HCCL_INFO("[aclstub][aclrtSetCurrentContext]deviceId: %u", curr_dev_id);
    return ACL_SUCCESS;
}

aclError aclrtGetDevicesTopo(uint32_t devId, uint32_t otherDevId, uint64_t* value)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream* stream)
{
    HcclSim::SimNpu& npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    *stream = npu.AllocMainStream();
    return ACL_SUCCESS;
}

aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

int rtModelFake = 0;
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* modelRI)
{
    *modelRI = &rtModelFake;
    return ACL_SUCCESS;
}

// 用例二进制主流程依赖南向桩函数
aclError aclrtSetDevice(int32_t deviceId)
{
    curr_dev_id = deviceId;
    HCCL_INFO("[aclstub][aclrtSetDevice]deviceId: %d", deviceId);
    return ACL_SUCCESS;
}

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    if (kernelName == nullptr || funcHandle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *funcHandle = std::strcmp(kernelName, "HcclLaunchAicpuKernel") == 0 ? &g_aicpuKernelHandle : nullptr;
    return ACL_SUCCESS;
}

// 可变长参数定义
struct ArgsBuffer {
    void* data;
    uint64_t size;
};

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle* argsHandle)
{
    if (argsHandle == nullptr) {
        HCCL_ERROR("[aclrtKernelArgsInit] invalid input argsHandle");
        return ACL_ERROR_INVALID_PARAM;
    }

    // 仅申请ArgsBuffer，存放OpParam的空间在aclrtKernelArgsAppend时分配
    ArgsBuffer* buffer = (ArgsBuffer*)malloc(sizeof(ArgsBuffer));
    if (buffer == nullptr) {
        HCCL_ERROR("[aclrtKernelArgsInit] malloc ArgsBuffer failed");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    buffer->data = nullptr;
    buffer->size = 0;
    *argsHandle = reinterpret_cast<void*>(buffer);

    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void* param, size_t paramSize, aclrtParamHandle* paramHandle)
{
    if (argsHandle == nullptr || param == nullptr || paramSize == 0) {
        HCCL_ERROR("[aclrtKernelArgsAppend] invalid input param");
        return ACL_ERROR_INVALID_PARAM;
    }

    // 按照OpParam的真实大小paramSize分配内存
    ArgsBuffer* buffer = reinterpret_cast<ArgsBuffer*>(argsHandle);
    buffer->data = malloc(paramSize);
    buffer->size = paramSize;
    if (buffer->data == nullptr) {
        HCCL_ERROR("[aclrtKernelArgsAppend] malloc buffer->data failed");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    memcpy(buffer->data, param, paramSize);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    // LLT模式下无需处理，直接返回成功
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t* notifyId)
{
    if (notify == nullptr || notifyId == nullptr) {
        HCCL_ERROR("[aclrtGetNotifyId] invalid input notify or notifyId");
        return ACL_ERROR_INVALID_PARAM;
    }

    HcclSim::SimNotify* simNotify = reinterpret_cast<HcclSim::SimNotify*>(notify);
    if (simNotify != nullptr) {
        *notifyId = simNotify->GetNotifyId();
    }
    return ACL_SUCCESS;
}

aclError aclrtCreateNotify(aclrtNotify* notify, uint64_t flag)
{
    HcclSim::SimNpu& npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    *notify = npu.AllocNotify();
    return ACL_SUCCESS;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    HcclSim::SimNotify* simNotify = reinterpret_cast<HcclSim::SimNotify*>(notify);
    if (simNotify == nullptr) {
        HCCL_ERROR("[aclrtWaitAndResetNotify] invalid input notify");
        return ACL_ERROR_INVALID_PARAM;
    }

    auto task = std::make_shared<HcclSim::TaskStubLocalWaitFrom>(simNotify->GetNotifyId());
    auto npuPos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curr_dev_id);
    HcclSim::SimTaskQueue::Global()->AppendTask(npuPos, reinterpret_cast<HcclSim::SimStream*>(stream), task);
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    // LLT模式下不从文件加载句柄, 桩函数直接返回成功
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg,
    aclrtArgsHandle argsHandle, void* reserve)
{
    if (argsHandle == nullptr || stream == nullptr) {
        HCCL_ERROR("[aclrtLaunchKernelWithConfig] invalid input argsHandle or stream");
        return ACL_ERROR_INVALID_PARAM;
    }

    ArgsBuffer* buffer = reinterpret_cast<ArgsBuffer*>(argsHandle);
    OpParam* param = reinterpret_cast<OpParam*>(buffer->data);
    HcclLaunchAicpuKernel(param);

    if (argsHandle != nullptr) {
        // argsHandle内存由aclrtKernelArgsInit开辟此处释放
        free(buffer->data);
        buffer->data = nullptr;
        free(argsHandle);
        argsHandle = nullptr;
    }
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs,
    size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    if (funcHandle != &g_aicpuKernelHandle) {
        HCCL_WARNING("[%s] not support.", __func__);
        return ACL_SUCCESS;
    }
    if (hostArgs == nullptr || stream == nullptr || argsSize < sizeof(OpParam)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const auto* param = static_cast<const OpParam*>(hostArgs);
    if (param->varMemSize > argsSize - sizeof(OpParam)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    // Match Runtime's argument copy: the kernel may restore pointers in its own OpParam copy.
    std::unique_ptr<void, decltype(&free)> argsCopy(malloc(argsSize), &free);
    if (argsCopy == nullptr) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    if (memcpy_s(argsCopy.get(), argsSize, hostArgs, argsSize) != EOK) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HcclLaunchAicpuKernel(static_cast<OpParam*>(argsCopy.get()));
    return ACL_SUCCESS;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    HcclSim::SimNotify* simNotify = reinterpret_cast<HcclSim::SimNotify*>(notify);
    if (simNotify == nullptr) {
        HCCL_ERROR("[aclrtWaitAndResetNotify] invalid input notify");
        return ACL_ERROR_INVALID_PARAM;
    }

    auto task = std::make_shared<HcclSim::TaskStubLocalPostTo>(simNotify->GetNotifyId());
    auto npuPos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curr_dev_id);
    HcclSim::SimTaskQueue::Global()->AppendTask(npuPos, reinterpret_cast<HcclSim::SimStream*>(stream), task);
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* value)
{
// 老CANN的acl_rt.h没有这个枚举, 探测与它同批引入的宏, 口径同adapter_acl.h
#ifdef ACL_DEVICE_FORM_FACTOR_POD
    // 本函数其余分支不回写*value。调用方普遍把出参初始化为0, 而ACL_DEVICE_FORM_FACTOR_POD恰好是0,
    // 不显式给出非POD取值的话, ST会把每个用例都建模成POD机型
    if (attr == ACL_DEV_ATTR_DEVICE_FORM_FACTOR) {
        *value = ACL_DEVICE_FORM_FACTOR_A_X;
        return ACL_SUCCESS;
    }
#endif
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetLogicDevIdByPhyDevId(int32_t phyDevId, int32_t* const logicDevId)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t* const logicDevId)
{
    // SimWorld不区分user/logic设备号, aclrtGetDevice直接返回curr_dev_id, 这里原样透传
    *logicDevId = userDevId;
    return ACL_SUCCESS;
}

aclError aclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t* const phyDevId)
{
    auto npuPos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curr_dev_id);
    *phyDevId = npuPos.phyId;
    return ACL_SUCCESS;
}

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return RT_ERROR_NONE;
}

aclError aclsysGetVersionNum(char* pkgNname, int32_t* versionNum)
{
    const char* versionEnv = getenv("ST_HCOMM_VERSION");
    if (versionEnv != nullptr && versionEnv[0] != '\0') {
        *versionNum = static_cast<int32_t>(strtol(versionEnv, nullptr, 10));
    } else {
        *versionNum = 90000009;
    }
    return ACL_SUCCESS;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode* mode) { return ACL_SUCCESS; }

aclError aclrtGetOpTimeOutInterval(uint64_t* interval) { return ACL_SUCCESS; }

uint32_t aclrtGetTaskIdFromExceptionInfo(const aclrtExceptionInfo* info)
{
    if (info == nullptr) {
        return 0;
    }
    const rtExceptionInfo_t* exceptionInfo = reinterpret_cast<const rtExceptionInfo_t*>(info);
    return exceptionInfo->taskid;
}

uint32_t aclrtGetStreamIdFromExceptionInfo(const aclrtExceptionInfo* info)
{
    if (info == nullptr) {
        return 0;
    }
    const rtExceptionInfo_t* exceptionInfo = reinterpret_cast<const rtExceptionInfo_t*>(info);
    return exceptionInfo->streamid;
}

uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo* info)
{
    if (info == nullptr) {
        return 0;
    }
    const rtExceptionInfo_t* exceptionInfo = reinterpret_cast<const rtExceptionInfo_t*>(info);
    return exceptionInfo->deviceid;
}

uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo* info)
{
    if (info == nullptr) {
        return 0;
    }
    const rtExceptionInfo_t* exceptionInfo = reinterpret_cast<const rtExceptionInfo_t*>(info);
    return exceptionInfo->retcode;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t* taskId, uint32_t* streamId)
{
    if (taskId == nullptr || streamId == nullptr) {
        return RT_ERROR_NONE;
    }
    *taskId = 0;
    *streamId = 0;
    return RT_ERROR_NONE;
}

aclError aclrtGetThreadLastTaskIdAndStreamId(uint32_t* taskId, uint32_t* streamId)
{
    if (taskId == nullptr || streamId == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *taskId = 0;
    *streamId = 0;
    return ACL_SUCCESS;
}

aclError aclrtCreateEventExWithFlag(aclrtEvent* event, uint32_t flag)
{
    if (event == nullptr) {
        HCCL_ERROR("[aclrtCreateEventExWithFlag] invalid input event");
        return ACL_ERROR_INVALID_PARAM;
    }
    *event = malloc(sizeof(int));
    if (*event == nullptr) {
        HCCL_ERROR("[aclrtCreateEventExWithFlag] malloc event failed");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    return ACL_SUCCESS;
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    if (event != nullptr) {
        free(event);
    }
    return ACL_SUCCESS;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
