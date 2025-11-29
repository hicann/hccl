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
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "runtime/base.h"
#include "runtime/stream.h"
#include "hccl/hccl_types.h"
#include "sim_world.h"
#include "sim_stream.h"
#include "hccl_sim_pub.h"
#include "log.h"

using namespace hccl;
thread_local uint32_t curr_dev_id = UINT32_MAX;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
aclError aclrtFreeHost(void *hostPtr)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtMallocHost(void **hostPtr, size_t size)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

// 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    u32 memType = static_cast<u32>(policy);
    HcclSim::SimNpu &simNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    if (memType == BUFFER_INPUT_MARK) {
        *devPtr = reinterpret_cast<void *>(simNpu.AllocMemory(BufferType::INPUT, size));
    } else if (memType == BUFFER_OUTPUT_MARK) {
        *devPtr = reinterpret_cast<void *>(simNpu.AllocMemory(BufferType::OUTPUT, size));
    }
    return ACL_SUCCESS;
}

aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId_)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

const std::map<DevType, std::string> DEV_VERSION_MAP = {
    {DevType::DEV_TYPE_910B, "Ascend910B1"},
    {DevType::DEV_TYPE_910_93, "Ascend910_9391"}
};

const char *aclrtGetSocName()
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

HcclResult hrtGetDeviceType(DevType &devType)
{
    auto npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    devType = npu.GetDevType();
    return HCCL_SUCCESS;
}

aclError aclrtGetDevice(int32_t* device )
{
    *device = curr_dev_id;
    return ACL_SUCCESS;
}

rtError_t rtStreamGetCaptureInfo(rtStream_t stm, rtStreamCaptureStatus *status, rtModel_t *captureMdl)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return RT_ERROR_NONE;
}

aclError aclrtGetDevicesTopo(uint32_t devId, uint32_t otherDevId, uint64_t *value)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream)
{
    HcclSim::SimNpu& npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curr_dev_id);
    *stream = npu.AllocMainStream();
    return ACL_SUCCESS;
}

int rtModelFake = 0;
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI)
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

HcclResult hrtGetDeviceIndexByPhyId(uint32_t devicePhyId, uint32_t &deviceLogicId)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName,
    aclrtFuncHandle *funcHandle)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
    aclrtParamHandle *paramHandle)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtCreateNotify(aclrtNotify *notify, uint64_t flag)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options,
    aclrtBinHandle *binHandle)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim,
    aclrtStream stream, aclrtLaunchKernelCfg *cfg,
    aclrtArgsHandle argsHandle, void *reserve)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetLogicDevIdByPhyDevId(int32_t phyDevId, int32_t *const logicDevId)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus