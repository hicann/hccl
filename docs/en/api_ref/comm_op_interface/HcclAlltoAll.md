# HcclAlltoAll

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T07:45:36.277Z pushedAt=2026-09-18T08:03:58.114Z -->

## Applicable Products

<!-- npu="950" id1 -->
- 950PR/950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

The operation API for the collective communication operator AlltoAll, which sends data of the same size to all ranks in the communicator and receives data of the same size from all ranks.

![alltoall](figures/alltoall.png)

The AlltoAll operation splits the input data into a specific number of blocks along a specific dimension, sends them to other ranks in order, and simultaneously receives input data from other ranks, concatenating the data along the specific dimension in order.

## Prototype

```c
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| sendCount | Input | Amount of data sent to each rank. |
| sendType | Input | Data type of the sent data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types).|
| recvBuf | Output | Address of the destination data buffer where the collective communication result is output.<br>recvBuf cannot have the same address as sendBuf, and their memory range cannot overlap. |
| recvCount | Input | Amount of data received from each rank, which must be the same as sendCount. |
| recvType | Input | Data type of the received data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type, which must be the same as sendType.|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### Supported Data Types

<!-- npu="950" id10 -->
- For 950PR/950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.
<!-- end id10 -->
<!-- npu="A3" id11 -->
- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id12 -->
<!-- npu="910" id13 -->
- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id13 -->
<!-- npu="310p" id6 -->
- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id6 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | The passed pointer parameter is null, for example, comm, sendBuf, recvBuf, or stream is nullptr. |
| HCCL_E_PARA | The passed parameter is invalid, for example, sendCount is inconsistent with recvCount, sendType is inconsistent with recvType, or sendBuf has the same address as recvBuf. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, sendType/recvType is invalid or not supported by the current model. |
| HCCL_E_INTERNAL | Internal error. |
| HCCL_E_UNAVAIL | The resource is unavailable, e.g., an exception occurs during CCU resource reuse (reused threads are empty, or the thread count exceeds the limit). |
| HCCL_E_RUNTIME | Failed to call the underlying runtime API (e.g., memory copy or memory reset). |
| HCCL_E_MEMORY | Memory allocation or operation failed. |

## Constraints

- The sendCount, sendType, recvCount, and recvType of all ranks must be the same.
- The performance of the AlltoAll operation depends on the buffer size for data sharing between NPUs. When the communication data volume exceeds the buffer size, performance will degrade significantly. If the AlltoAll communication data volume in your service is large, you are advised to configure the environment variable [HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md) to appropriately increase the buffer size to improve communication performance.
<!-- npu="910" id7 -->
- For Atlas training products, the AlltoAll communicator must meet the following constraints:

    For a single server, 1p and 2p communicators must be within the same cluster (devices 0-3 and devices 4-7 in a server each form a cluster). For single-server 4p and 8p and multi-server communicators, ranks must be organized by cluster as the basic unit, and the cluster selection across servers must be consistent.

- For Atlas training products, in single-server use cases, the NIC status must be "up"; otherwise, this API will fail to execute.
<!-- end id7 -->
<!-- npu="310p" id14 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.
<!-- end id14 -->
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

For the Atlas 300I Duo Inference Card, initialize the communicator before allocating device memory. See [Constraints and Limitations](../../user_guide/constraints.md#atlas-300i-duo-inference-card).

The following application function shows the call order. Initialize the ACL runtime before calling it. `rankId` is the communicator rank and `deviceId` is the local logical device ID. Rank 0 obtains `rootInfo` using HcclGetRootInfo and distributes the same value to all ranks through the application's out-of-band channel before passing it to this function. Configure `rankSize` and other arguments according to the interface's product and networking constraints.

```cpp
#include <cstdint>
#include "hccl/hccl.h"

void ExampleHcclAlltoAll(uint32_t rankSize, uint32_t rankId, uint32_t deviceId, const HcclRootInfo& rootInfo)
{
    // Select the local logical device and initialize the communicator.
    aclrtSetDevice(deviceId);
    HcclComm hcclComm = nullptr;
    HcclCommInitRootInfo(rankSize, &rootInfo, rankId, &hcclComm);

    // Create the stream.
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // Allocate device memory for the collective communication operation.
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = rankSize;
    size_t mallocSize = count * sizeof(float);
    aclrtMalloc((void**)&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    aclrtMalloc((void**)&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

    // Execute AlltoAll to send the same amount of data to all ranks in the communicator and receive the same amount of
    // data from all ranks.
    size_t perCount = count / rankSize;
    HcclAlltoAll(sendBuf, perCount, HCCL_DATA_TYPE_FP32, recvBuf, perCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
    // Block until the collective communication tasks in the task stream complete.
    aclrtSynchronizeStream(stream);

    // Release resources.
    aclrtFree(sendBuf);         // Free memory on the device.
    aclrtFree(recvBuf);         // Free memory on the device.
    aclrtDestroyStream(stream); // Destroy the task stream.
    HcclCommDestroy(hcclComm);  // Destroy the communicator.
}
```
