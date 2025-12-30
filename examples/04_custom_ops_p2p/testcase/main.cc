/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>

#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_p2p.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclRootInfo *rootInfo;
    int32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    int32_t device = ctx->device;
    uint32_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);

    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(device));

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 执行 Send/Recv 操作，0/2/4/6卡发送数据，1/3/5/7接收数据
    // HcclSend 与 HcclRecv 接口采用同步调用方式，且必须配对使用
    if (device % 2 == 0) {
        // 申请 Device 内存用于存放输入数据
        ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_FIRST));
        // 申请 Host 内存用于设置输入数据，将内容初始化为 DeviceId
        void *hostBuf = nullptr;
        ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
        float *tmpHostBuf = static_cast<float *>(hostBuf);
        for (uint32_t i = 0; i < count; ++i) {
            tmpHostBuf[i] = static_cast<float>(device);
        }
        ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLCHECK(aclrtFreeHost(hostBuf));
        // 执行 Send 操作
        HCCLCHECK(HcclSendCustom(sendBuf, count, HCCL_DATA_TYPE_FP32, device + 1, hcclComm, stream));
    } else {
        // 申请 Device 内存用于接收数据
        ACLCHECK(aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_FIRST));
        // 执行 Recv 操作
        HCCLCHECK(HcclRecvCustom(recvBuf, count, HCCL_DATA_TYPE_FP32, device - 1, hcclComm, stream));
    }

    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    if (device % 2 != 0) {
        std::this_thread::sleep_for(std::chrono::seconds(device));
        void *resultHostBuf;
        ACLCHECK(aclrtMallocHost(&resultHostBuf, mallocSize));
        ACLCHECK(aclrtMemcpy(resultHostBuf, mallocSize, recvBuf, mallocSize, ACL_MEMCPY_DEVICE_TO_HOST));
        float *tmpResultBuf = static_cast<float *>(resultHostBuf);
        std::cout << "rankId: " << device << ", output: [";
        for (uint32_t i = 0; i < count; ++i) {
            std::cout << " " << tmpResultBuf[i];
        }
        std::cout << " ]" << std::endl;
        ACLCHECK(aclrtFreeHost(resultHostBuf));
    }

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm));  // 销毁通信域
    if (sendBuf) {
        ACLCHECK(aclrtFree(sendBuf));      // 释放 Device 侧内存
    }
    if (recvBuf) {
        ACLCHECK(aclrtFree(recvBuf));      // 释放 Device 侧内存
    }
    ACLCHECK(aclrtDestroyStream(stream));  // 销毁任务流
    ACLCHECK(aclrtResetDevice(device));    // 重置设备
    return 0;
}

int main()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 查询设备数量
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成 Root 节点信息，各线程使用同一份 RootInfo
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 启动线程执行集合通信操作
    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    // 释放资源
    ACLCHECK(aclrtFreeHost(rootInfoBuf));  // 释放 Host 内存
    ACLCHECK(aclFinalize());               // 设备去初始化
    return 0;
}
