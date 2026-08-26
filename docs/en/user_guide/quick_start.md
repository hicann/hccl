# Quick Start

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T04:00:55.571Z pushedAt=2026-08-13T10:26:42.496Z -->

This section uses the AllReduce operator as an example to introduce its usage in single-operator execution mode, helping users quickly experience collective communication functionality.

## AllReduce Operator Introduction

The AllReduce operation performs a reduction operation (supporting sum, prod, max, and min) on the input data of all nodes within a communicator, and then sends the result to the output buffer of all nodes.

![AllReduce operator diagram](figures/allreduce.png)

Note: Each rank can have only one input.

## Sample Introduction

You can click [Sample Link](https://gitcode.com/cann/hcomm/tree/9.1.0/examples/01_communicators/03_one_device_per_pthread) to obtain the complete sample code. This sample creates a communicator based on root node information and manages one AI Server in a single process, where each NPU device is managed by one thread. The sample mainly includes the following functional points:

- Device detection: queries the number of available devices through the `aclrtGetDeviceCount()` API.

- Uses rank0 as the root node and generates the rootInfo identification information of the root node through the `HcclGetRootInfo()` API.

- Based on rootInfo, initializes the communicator in each thread through the `HcclCommInitRootInfo()` API.

- Call the `HcclAllReduce( )` API to add the input data of all ranks in the communicator, send the result to all nodes, and print the result.

## Compilation and Execution

Run the following commands in the sample code directory:

```bash
make
make test
```

## Result Analysis

The data of each rank is initialized to 0–7. After the AllReduce operation, the result of each rank is the sum of the data at the corresponding positions of all ranks (the sum of data from eight ranks).

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

## Key Code Analysis

1. Use rank0 as the root node and generate rootInfo identification information, which mainly includes the device IP, device ID, and other details. This information must be broadcast to all ranks in the cluster for communicator initialization.

    ```c
    int rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // Generate root node information. All threads use the same rootInfo.
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));
    ```

2. Allocate memory and construct input data.

    ```c
    // Set the device operated by the current thread.
    ACLCHECK(aclrtSetDevice(ctx->device));
    
    // Allocate device memory for the collective communication operation.
    size_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);
    ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    
    // Allocate Host memory for storing input data, and initialize the content to 0 to 7.
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
    float *tmpHostBuff = static_cast<float *>(hostBuf);
    for (uint32_t i = 0; i < count; ++i) {
        tmpHostBuff[i] = static_cast<float>(i);
    }
    
    // Copy the Host-side input data to the device side.
    ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ```

3. Initialize the communicator.

    ```c
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, ctx->device, &hcclComm));
    ```

4. Execute the AllReduce collective communication operator.

    ```c
    // Create a task stream.
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));
    
    // Execute AllReduce to sum the sendBuf values of all ranks in the communicator, and then send the result to the recvBuf of all ranks.
    HCCLCHECK(HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream));
    // Block and wait for the collective communication task in the task stream to complete.
    ACLCHECK(aclrtSynchronizeStream(stream));
    ```

5. Release resources.

    ```c
    ACLCHECK(aclrtFree(sendBuf));          // Free memory on the device side.
    ACLCHECK(aclrtFree(recvBuf));          // Free the device-side memory.
    ACLCHECK(aclrtFreeHost(hostBuf));      // Free the host-side memory.
    ACLCHECK(aclrtDestroyStream(stream));  // Destroy the task stream.
    HCCLCHECK(HcclCommDestroy(hcclComm));  // Destroy the communicator.
    ACLCHECK(aclFinalize());               // Deinitialize the device.
    ```
