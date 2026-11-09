# HcclBroadcast

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:04:52.005Z pushedAt=2026-09-18T08:03:58.125Z -->

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
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

The operation API of the collective communication operator Broadcast, which broadcasts data from the root node in the communicator to other ranks.

![broadcast](figures/broadcast.png)

## Prototype

```c
HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| buf | Input/Output | Address of the data buffer. For the root node, it is the address of the source data buffer; for non-root nodes, it is the address of the buffer for receiving data. |
| count | Input | Number of data units involved in the broadcast operation. For example, `count` is `1` if only one int32 data unit is involved. |
| dataType | Input | Data type of the broadcast operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Data Type Description](#supported-data-types). |
| root | Input | Rank ID of the broadcast root. |
| comm | Input | Communicator in which the collective communication operation is performed. |
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

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | A passed pointer parameter is null. For example, comm, buf, or stream is nullptr. |
| HCCL_E_PARA | A passed parameter is invalid. For example, `count` exceeds the upper limit, or root is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported. For example, dataType is invalid or not supported by the current model. |
| HCCL_E_INTERNAL | Internal error. |
| HCCL_E_UNAVAIL | The resource is unavailable, e.g., an exception occurs during CCU resource reuse (reused threads are empty, or the thread count exceeds the limit). |
| HCCL_E_RUNTIME | Failed to call the underlying runtime API (e.g., memory copy or memory reset). |
| HCCL_E_MEMORY | Memory allocation or operation failed. |

## Constraints

- All ranks must have the same count, dataType, and root.
- There can be only one root node globally.
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

The following application function shows the call order. Initialize the ACL runtime before calling it. `rankId` is the communicator rank and `deviceId` is the local logical device ID. Rank 0 obtains `rootInfo` using HcclGetRootInfo and distributes the same value to all ranks through the application's out-of-band channel before passing it to this function. Configure `rankSize` and other arguments according to the interface's product and networking constraints. On the root rank, `hostBuf` points to at least 8 float elements and `rootRank` names a valid communicator rank.

```cpp
#include <cstdint>
#include "hccl/hccl.h"

void ExampleHcclBroadcast(
    uint32_t rankSize, uint32_t rankId, uint32_t deviceId, const HcclRootInfo& rootInfo, uint32_t rootRank,
    const float* hostBuf)
{
    // Select the local logical device and initialize the communicator.
    aclrtSetDevice(deviceId);
    HcclComm hcclComm = nullptr;
    HcclCommInitRootInfo(rankSize, &rootInfo, rankId, &hcclComm);

    // Create the stream.
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // Allocate device memory for the collective communication operation.
    void* buf
        = nullptr; // For the root node, it is the data source; for non-root nodes, it is the buffer for receiving data.
    uint64_t count = 8; // Number of data elements involved in the broadcast operation.
    size_t mallocSize = count * sizeof(float);
    aclrtMalloc(&buf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

    // Construct the input data on the root node.
    if (rankId == rootRank) {
        aclrtMemcpy(buf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // Broadcast the data of the root node in the communicator to other ranks.
    HcclBroadcast(buf, count, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
    // Block and wait for the collective communication task in the task stream to complete.
    aclrtSynchronizeStream(stream);

    // Release resources.
    aclrtFree(buf);             // Free memory on the device.
    aclrtDestroyStream(stream); // Destroy the task stream.
    HcclCommDestroy(hcclComm);  // Destroy the communicator.
}
```
