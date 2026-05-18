# 快速入门

本节以AllReduce算子为例，介绍其在单算子执行模式下的使用方式，帮助用户快速体验集合通信功能。

## AllReduce算子介绍

AllReduce操作是将通信域内所有节点的输入数据进行归约操作后（支持sum、prod、max、min），再把结果发送到所有节点的输出buffer。

![AllReduce算子图示](figures/allreduce.png)

注意：每个rank只能有一个输入。

## 样例介绍

用户可以点击[样例链接](https://gitcode.com/cann/hcomm/tree/master/examples/01_communicators/03_one_device_per_pthread)获取完整样例代码，该样例基于root节点信息创建通信域，在一个进程中管理一个AI Server，其中每个NPU设备由一个线程进行管理，主要包含以下功能点：

- 设备检测，通过aclrtGetDeviceCount( )接口查询可用设备数量。
- 将rank0作为root节点，通过HcclGetRootInfo( )接口生成root节点的rootInfo标识信息。

- 基于rootInfo，在每个线程中通过HcclCommInitRootInfo( )接口初始化通信域。
- 调用HcclAllReduce( ) 接口，将通信域内所有rank的输入数据进行相加后，再把结果发送到所有节点，并打印结果。

## 编译运行

在本样例代码目录下执行如下命令：

```bash
make
make test
```

## 结果解析

每个rank的数据初始化为0~7，经过AllReduce操作后，每个rank的结果是所有rank对应位置数据的和（8个rank的数据相加）。

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 1, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 2, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 3, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 4, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 5, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 6, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 7, output: [ 0 8 16 24 32 40 48 56 ]
```

## 关键代码解析

1. 将rank0作为root节点，生成rootInfo标识信息，主要包含：Device IP、Device ID等信息。此信息需广播至集群内所有rank用来初始化通信域。

    ```c
    int rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成root节点信息，各线程使用同一份rootInfo
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));
    ```

2. 申请内存，构造输入数据。

    ```c
    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(ctx->device));
    
    // 申请集合通信操作的Device内存
    size_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);
    ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    
    // 申请 Host 内存用于存放输入数据，并将内容初始化为：0~7
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
    float *tmpHostBuff = static_cast<float *>(hostBuf);
    for (uint32_t i = 0; i < count; ++i) {
        tmpHostBuff[i] = static_cast<float>(i);
    }
    
    // 将Host侧输入数据拷贝到Device侧
    ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ```

3. 初始化通信域。

    ```c
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, ctx->device, &hcclComm));
    ```

4. 执行AllReduce集合通信算子。

    ```c
    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));
    
    // 执行 AllReduce，将通信域内所有节点的 sendBuf 进行相加后，再把结果发送到所有节点的 recvBuf
    HCCLCHECK(HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream));
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));
    ```

5. 释放资源。

    ```c
    ACLCHECK(aclrtFree(sendBuf));          // 释放 Device 侧内存
    ACLCHECK(aclrtFree(recvBuf));          // 释放 Device 侧内存
    ACLCHECK(aclrtFreeHost(hostBuf));      // 释放 Host 侧内存
    ACLCHECK(aclrtDestroyStream(stream));  // 销毁任务流
    HCCLCHECK(HcclCommDestroy(hcclComm));  // 销毁通信域
    ACLCHECK(aclFinalize());               // 设备去初始化
    ```
