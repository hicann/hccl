# HCCL_NPU_SOCKET_PORT_RANGE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:48:10.546Z pushedAt=2026-09-18T08:03:58.213Z -->

## Description

When a communicator is created based on root node information, you can use this environment variable to set the communication port used by HCCL on the NPU.

This environment variable can be set as a specific port, a port range, or the string `auto`.

- If a specific port number or port range is used, the number of planned ports should be no less than the number of HCCL processes on a single NPU. The port number ranges from 1 to 65535, and you must ensure that the specified ports are not occupied by other processes. Note that ports 1 to 1023 are reserved by the system and should not be used.

    Port numbers and port ranges can be used in combination, separated by commas (,). However, the port numbers or port ranges between commas must not overlap. For details, see [Configuration Example](#configuration-example).

- `auto` means the NPU port number used by HCCL is dynamically allocated by the operating system.
- If this environment variable is not configured, the default communication port used by HCCL on the NPU is 16666.

## Configuration Example

```bash
// Method 1: Configure as a port range.
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"
// Method 2: Use a specific port number together with a port range, separated by commas (,).
export HCCL_NPU_SOCKET_PORT_RANGE="61000,61050-61100,61200-61210"
// Method 3: Specify specific port numbers, separated by commas (,).
export HCCL_NPU_SOCKET_PORT_RANGE="57000,57005,57007,58008,58100,58105,58107,58108"
// Method 4: The OS dynamically allocates port numbers.
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
```

## Constraints

- In multi-device use cases, this environment variable works if it's set for all devices in the same communicator.
- In single-device multi-process use cases (multiple processes share one NPU), set this environment variable to avoid service failures due to port conflicts. However, note that running multiple processes will have a certain impact on resource overhead and communication performance.

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
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
