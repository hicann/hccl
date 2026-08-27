/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <hccl_custom_allgather.h>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::seconds;

#define ACLCHECK(expr)                                                                                  \
    do {                                                                                                \
        auto _ret = (expr); /* 执行一次并保存结果 */                                           \
        if (_ret != ACL_SUCCESS) {                                                                      \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                                \
        }                                                                                               \
    } while (0)

#define HCCLCHECK(expr)                                                                                  \
    do {                                                                                                 \
        auto _ret = (expr); /* 执行一次并保存结果 */                                            \
        if (_ret != HCCL_SUCCESS) {                                                                      \
            printf("[ERROR] hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                                 \
        }                                                                                                \
    } while (0)

inline void BuildLogString(std::ostringstream& oss) {}

template <typename T, typename... Args>
inline void BuildLogString(std::ostringstream& oss, const T& first, const Args&... args)
{
    oss << first;
    BuildLogString(oss, args...);
}

template <typename... Args>
void Log(int rank, const Args&... args)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    std::ostringstream oss;

    oss << "[" << tv.tv_sec << "." << std::setfill('0') << std::setw(6) << tv.tv_usec << "] [Rank " << rank << "] ";

    BuildLogString(oss, args...);

    std::cout << oss.str() << std::endl;
}

int PrepareData(
    int rank, uint64_t count, size_t sendBytes, size_t recvBytes, aclrtStream& stream, void*& sendBuf, void*& recvBuf)
{
    ACLCHECK(aclrtCreateStream(&stream));
    ACLCHECK(aclrtMalloc(&sendBuf, sendBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<float> hostSend(count, (float)rank);
    ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(recvBuf, recvBytes, 0, recvBytes));
    Log(rank, "Buffers allocated and initialized");
    return 0;
}

int VerifyResult(int rank, int rankSize, uint64_t count, size_t recvBytes, void* recvBuf)
{
    std::vector<float> hostRecv(count * rankSize);
    ACLCHECK(aclrtMemcpy(hostRecv.data(), recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    for (int r = 0; r < rankSize; r++) {
        for (uint64_t i = 0; i < count; i++) {
            float val = hostRecv[r * count + i];
            if (std::abs(val - (float)r) > 1e-5) {
                Log(rank, "Error at rank ", r, " offset ", i, ": expected ", r, ", got ", val);
                return -1;
            }
        }
    }
    Log(rank, "VerifyResult Passed!");
    return 0;
}

struct ThreadContext {
    HcclRootInfo* rootInfo;
    uint32_t rank;
    uint32_t rankSize;
    uint32_t device;
    uint32_t devCount;
    uint32_t dataLen;
};

int Sample(void* arg)
{
    ThreadContext* ctx = (ThreadContext*)arg;
    int rank = ctx->rank;
    int rankSize = ctx->rankSize;
    int device = rank % ctx->devCount;
    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));
    Log(rank, "HCCL set device[", device, "]");

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(rankSize, ctx->rootInfo, device, &hcclComm));
    Log(rank, "HCCL Comm Initialized");

    uint64_t dataLen = ctx->dataLen;
    size_t sendBytes = dataLen * sizeof(float);
    size_t recvBytes = dataLen * rankSize * sizeof(float);
    size_t recvSize = dataLen * rankSize;

    aclrtStream stream = nullptr;
    void *sendBuf = nullptr, *recvBuf = nullptr;

    HCCLCHECK(PrepareData(rank, dataLen, sendBytes, recvBytes, stream, sendBuf, recvBuf));

    auto start = high_resolution_clock::now();
    // 执行 AllGather，将通信域内所有 rank 的 sendBuf 按照 rank_id 顺序拼接起来，再将结果发送到所有 rank 的 recvBuf
    HCCLCHECK(HcclAllGatherCustom(sendBuf, recvBuf, dataLen, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));
    auto end = high_resolution_clock::now();

    // 计算差值，转毫秒
    auto duration_ms = duration_cast<milliseconds>(end - start);
    std::cout << "rank" << rank << " dataLen=" << ctx->dataLen << " time=" << duration_ms.count() << " ms\n";

    // 接收数据校验
    HCCLCHECK(VerifyResult(rank, rankSize, dataLen, recvBytes, recvBuf));

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    std::this_thread::sleep_for(std::chrono::seconds(1));
    void* resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, recvBytes));
    ACLCHECK(aclrtMemcpy(resultBuff, recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    float* tmpResBuff = static_cast<float*>(resultBuff);
    std::cout << "rankId: " << ctx->device << ", output: [";
    for (uint32_t i = 0; i < std::min(dataLen * rankSize, 64ul) /*recvSize*/; ++i) {
        std::cout << " " << tmpResBuff[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultBuff));

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm)); // 销毁通信域
    if (sendBuf) {
        ACLCHECK(aclrtFree(sendBuf)); // 释放 Device 侧内存
    }
    if (recvBuf) {
        ACLCHECK(aclrtFree(recvBuf)); // 释放 Device 侧内存
    }
    ACLCHECK(aclrtDestroyStream(stream)); // 销毁任务流
    ACLCHECK(aclrtResetDevice(device));   // 重置设备
    return 0;
}

#ifdef ENABLE_MPI
#include <mpi.h>
int main(int argc, char* argv[])
{
    uint32_t dataLen = argc > 1 ? atoi(argv[1]) : 1024;
    int rank = 0, size = 0;
    HcclComm hcclComm = nullptr;

    MPI_Init(&argc, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Log(rank, "MPI Initialized. World Size: ", size);

    ACLCHECK(aclInit(NULL));

    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    if (devCount == 0) {
        Log(rank, "Error: No devices found");
        return -1;
    }

    int deviceId = rank % devCount;
    Log(rank, "Device ", deviceId, " selected (Total devices: ", devCount, ")");
    ACLCHECK(aclrtSetDevice(deviceId));

    HcclRootInfo rootInfo;
    if (rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
        Log(rank, "Root info generated");
    }

    MPI_Bcast(&rootInfo, sizeof(HcclRootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

    ThreadContext args;
    args.rootInfo = &rootInfo;
    args.rank = rank;
    args.rankSize = size;
    args.device = rank;
    args.devCount = devCount;
    args.dataLen = dataLen;
    Sample((void*)&args);

    ACLCHECK(aclFinalize()); // 设备去初始化
    MPI_Finalize();

    return 0;
}
#else
int main(int argc, char* argv[])
{
    uint32_t rankSize = argc > 1 ? atoi(argv[1]) : 2;
    uint32_t dataLen = argc > 2 ? atoi(argv[2]) : 1024;
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 查询设备数量
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成 Root 节点信息，各线程使用同一份 RootInfo
    void* rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo* rootInfo = (HcclRootInfo*)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 启动线程执行集合通信操作
    std::vector<std::thread> threads(rankSize);
    std::vector<ThreadContext> args(rankSize);
    for (uint32_t i = 0; i < rankSize; i++) {
        args[i].rootInfo = rootInfo;
        args[i].rank = i;
        args[i].rankSize = rankSize;
        args[i].device = i;
        args[i].devCount = rankSize;
        args[i].dataLen = dataLen;
        threads[i] = std::thread(Sample, (void*)&args[i]);
    }
    for (uint32_t i = 0; i < rankSize; i++) {
        threads[i].join();
    }

    // 释放资源
    ACLCHECK(aclrtFreeHost(rootInfoBuf)); // 释放 Host 内存
    ACLCHECK(aclFinalize());              // 设备去初始化
    return 0;
}
#endif
