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

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

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
    int32_t rootRank;
    int32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    int32_t rootRank = ctx->rootRank;
    int32_t device = ctx->device;
    uint64_t sendCount = ctx->devCount;
    uint64_t recvCount = 1U;
    size_t sendSize = sendCount * sizeof(float);
    size_t recvSize = recvCount * sizeof(float);

    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(device));

    // 申请 Device 内存用于接收 Scatter 结果
    ACLCHECK(aclrtMalloc(&recvBuf, recvCount, ACL_MEM_MALLOC_HUGE_FIRST));
    // 在 Root 节点，申请 Device 内存用于存放发送数据，并将内容初始化为：0~7
    if (device == rootRank) {
        ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_FIRST));
        // 申请 Host 内存用于存放输入数据
        void *hostBuf = nullptr;
        ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
        float *tmpHostBuf = static_cast<float *>(hostBuf);
        for (uint32_t i = 0; i < sendCount; ++i) {
            tmpHostBuf[i] = static_cast<float>(i);
        }
        // 将 Host 侧输入数据拷贝到 Device 侧
        ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
        // 释放 Host 侧内存
        ACLCHECK(aclrtFreeHost(hostBuf));
    }

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 执行 Scatter，将通信域内 Root 节点的数据均分并散布至其他 Rank
    HCCLCHECK(HcclScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream));
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    std::this_thread::sleep_for(std::chrono::seconds(device));
    void *resultHostBuf;
    ACLCHECK(aclrtMallocHost(&resultHostBuf, recvSize));
    ACLCHECK(aclrtMemcpy(resultHostBuf, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float *tmpResultBuf = static_cast<float *>(resultHostBuf);
    std::cout << "rankId: " << device << ", output: [";
    for (uint32_t i = 0; i < recvCount; ++i) {
        std::cout << " " << tmpResultBuf[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultHostBuf));

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm));  // 销毁通信域
    if (sendBuf != nullptr) {
        ACLCHECK(aclrtFree(sendBuf));      // 释放 Device 侧内存
    }
    ACLCHECK(aclrtFree(recvBuf));          // 释放 Device 侧内存
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
        args[i].rootRank = rootRank;
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
