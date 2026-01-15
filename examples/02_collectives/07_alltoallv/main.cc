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
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint32_t rankSize = ctx->devCount;
    uint64_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);

    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // 申请集合通信操作的 Device 内存
    ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // 申请 Host 内存用于存放输入数据
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
    float *tmpHostBuff = static_cast<float *>(hostBuf);
    for (uint64_t i = 0; i < count; ++i) {
        tmpHostBuff[i] = static_cast<float>(device);
    }
    // 将 Host 侧输入数据拷贝到 Device 侧
    ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 释放 Host 侧内存
    ACLCHECK(aclrtFreeHost(hostBuf));

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(rankSize, ctx->rootInfo, device, &hcclComm));

    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 设置收发数据量，收发数据量相同
    std::vector<uint64_t> sendCounts(rankSize, 1);
    std::vector<uint64_t> recvCounts(rankSize, 1);
    std::vector<uint64_t> sdispls(rankSize);
    std::vector<uint64_t> rdispls(rankSize);
    for (size_t i = 0; i < rankSize; ++i) {
        sdispls[i] = i;
        rdispls[i] = i;
    }
    // 执行 AlltoAllV，向通信域内所有 rank 发送相同数据量的数据，并从所有 rank 接收相同数据量的数据，可定制数据量
    HCCLCHECK(HcclAlltoAllV(sendBuf,
        sendCounts.data(),
        sdispls.data(),
        HCCL_DATA_TYPE_FP32,
        recvBuf,
        recvCounts.data(),
        rdispls.data(),
        HCCL_DATA_TYPE_FP32,
        hcclComm,
        stream));
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    std::this_thread::sleep_for(std::chrono::seconds(device));
    void *resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, mallocSize));
    ACLCHECK(aclrtMemcpy(resultBuff, mallocSize, recvBuf, mallocSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float *tmpResBuff = static_cast<float *>(resultBuff);
    std::cout << "rankId: " << device << ", output: [";
    for (uint64_t i = 0; i < count; ++i) {
        std::cout << " " << tmpResBuff[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultBuff));

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm));  // 销毁通信域
    ACLCHECK(aclrtFree(sendBuf));          // 释放 Device 侧内存
    ACLCHECK(aclrtFree(recvBuf));          // 释放 Device 侧内存
    ACLCHECK(aclrtDestroyStream(stream));  // 销毁任务流
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

    int32_t rootRank = 0;
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
