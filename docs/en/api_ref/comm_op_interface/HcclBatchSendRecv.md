# HcclBatchSendRecv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:00:46.179Z pushedAt=2026-09-18T08:03:58.122Z -->

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

Asynchronous batch point-to-point communication API. A single API call can complete multiple send and receive tasks on the current rank. The send and receive operations on the current rank are asynchronous, meaning send and receive tasks do not block each other.

## Prototype

```c
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendRecvInfo | Input | Pointer to the list of Send/Recv tasks to be issued by this rank.<br>Type: HcclSendRecvItem. For details, see [HcclSendRecvItem](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclSendRecvItem.md). |
| itemNum | Input | Number of Send/Recv tasks of this rank. |
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by this rank. |

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
| HCCL_E_PTR | The passed pointer parameter is null, for example, comm, sendRecvInfo, or stream is nullptr. |
| HCCL_E_PARA | The passed parameter is invalid, for example, `count` of an item in sendRecvInfo exceeds the upper limit, or remoteRank is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, dataType is invalid or the current product model is not supported. |
| HCCL_E_INTERNAL | Internal error. |
| HCCL_E_UNAVAIL | The resource is unavailable, e.g., an exception occurs during CCU resource reuse (reused threads are empty, or the thread count exceeds the limit). |
| HCCL_E_RUNTIME | Failed to call the underlying runtime API (e.g., memory copy or memory reset). |
| HCCL_E_MEMORY | Memory allocation or operation failed. |

## Constraints

- "Asynchronous" means that the receive and send tasks on the same device are asynchronous and do not block each other. However, between devices, send and receive tasks are still synchronous. Therefore, inter-device send and receive tasks must be in one-to-one correspondence, just like HcclSend and HcclRecv.
- For Atlas A2 training products/Atlas A2 inference products, when using this API in large-scale clusters (ranksize > 500), the number of concurrent executions cannot exceed 3.
- For [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458), if a link establishment failure occurs between devices within a server (error code: EI0010), set the environment variable `HCCL_INTRA_ROCE_ENABLE` to `1` and `HCCL_INTRA_PCIE_ENABLE` to `0`, so that the RoCE loop is used for multi-device communication within the server (ensure that a RoCE NIC exists on the server and that the RDMA links between devices with send/recv relationships are reachable). Example environment variable configuration:

    ```bash
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_INTRA_PCIE_ENABLE=0
    ```
    
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

For the Atlas 300I Duo Inference Card, initialize the communicator before allocating device memory. See [Constraints and Limitations](../../user_guide/constraints.md#atlas-300i-duo-inference-card).

The following application function shows the call order. Initialize the ACL runtime before calling it. `rankId` is the communicator rank and `deviceId` is the local logical device ID. Rank 0 obtains `rootInfo` using HcclGetRootInfo and distributes the same value to all ranks through the application's out-of-band channel before passing it to this function. Configure `rankSize` and other arguments according to the interface's product and networking constraints.

```cpp
#include <cstdint>
#include "hccl/hccl.h"

void ExampleHcclBatchSendRecv(uint32_t rankSize, uint32_t rankId, uint32_t deviceId, const HcclRootInfo& rootInfo)
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
    uint64_t count = 4;
    size_t mallocSize = count * sizeof(float);
    aclrtMalloc((void**)&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    aclrtMalloc((void**)&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

    // Execute Send/Recv to send data to the next node and receive data from the previous node at the same time.
    // HcclBatchSendRecv can issue multiple Send/Recv tasks on this rank at the same time.
    uint32_t next = (rankId + 1) % rankSize;
    uint32_t prev = (rankId - 1 + rankSize) % rankSize;
    HcclSendRecvItem sendRecvInfo[2];
    sendRecvInfo[0] = HcclSendRecvItem{HCCL_SEND, sendBuf, count, HCCL_DATA_TYPE_FP32, next};
    sendRecvInfo[1] = HcclSendRecvItem{HCCL_RECV, recvBuf, count, HCCL_DATA_TYPE_FP32, prev};
    HcclBatchSendRecv(sendRecvInfo, 2, hcclComm, stream);

    // Block and wait for the collective communication tasks in the stream to complete.
    aclrtSynchronizeStream(stream);

    // Release resources.
    aclrtFree(sendBuf);         // Free device-side memory.
    aclrtFree(recvBuf);         // Free device-side memory.
    aclrtDestroyStream(stream); // Destroy the stream.
    HcclCommDestroy(hcclComm);  // Destroy the communicator.
}
```
