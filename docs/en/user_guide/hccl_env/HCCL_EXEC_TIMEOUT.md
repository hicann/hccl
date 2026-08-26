# HCCL_EXEC_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:55:21.431Z pushedAt=2026-08-06T06:44:03.570Z -->

## Function

During distributed training or inference, different device processes may encounter inconsistency in inter-device task execution (for example, only specific processes save checkpoint data). This environment variable controls the synchronization wait time for inter-device execution, within which each device process waits for other devices to complete communication synchronization.

- **For Ascend 950PR/Ascend 950DT**:

  - `AI_CPU` mode (see [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)): The unit is second. An integer value is recommended. The value ranges from 0 to 2147483647, and defaults to `1836`. Value `0` indicates no timeout.

  - `AIV` mode: The unit is second. The value ranges from 0 to 1091, and defaults to `1091`. Ten-millisecond precision is allowed (for example, to set a 50-millisecond timeout, use the value `0.05`). If set to 0 or a value exceeding 1091, it will be set as `1091`.

    In AIV mode, the actual effective timeout duration is interval\*N\*10<sup>-3</sup> milliseconds, where `interval` is the minimum operator timeout interval supported by hardware (which can be obtained through the aclrtGetOpTimeoutInterval API), in μs, and N is an integer ranging from 1 to 254. If the configured timeout duration is not equal to interval\*N\*10<sup>-3</sup> milliseconds, the latter is used.

- **For Atlas A3 training products/Atlas A3 inference products:**

  - `AI_CPU` and `AICPU_CacheDisable` modes (see [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)): The unit is second. An integer value is recommended. The value ranges from 0 to 2147483647, and defaults to `1836`. Value `0` indicates no timeout.

  - `AIV` mode: The unit is second. The value ranges from 0 to 1091, and defaults to `1091`. Ten-millisecond precision is allowed (for example, to set a 50-millisecond timeout, use the value `0.05`). If set to `0` or a value exceeding `1091`, it is set as `1091`.

    In AIV mode, the actual effective timeout duration is interval\*N\*10<sup>-3</sup> milliseconds, where interval is the minimum operator timeout interval supported by hardware (which can be obtained through the aclrtGetOpTimeoutInterval API), in μs, and N is an integer ranging from 1 to 254. If the configured timeout duration is not equal to interval\*N\*10<sup>-3</sup> milliseconds, the latter is used.

- **For Atlas A2 training products/Atlas A2 inference products:**

  - `HOST` and `HOST_TS` modes (see [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)): The unit is second. The value ranges from 0 to 2147483647, and defaults to `1836`. Integer-second precision is allowed. Value `0` indicates no timeout.

  - `AIV` mode: The unit is second. The value ranges from 0 to 1091, and defaults to `1091`. Ten-millisecond precision is allowed (for example, to set a 50-millisecond timeout, use the value `0.05`). If set to `0` or a value exceeding 1091, it will be set as `1091`.

    The actual effective timeout duration in AIV mode is interval\*N\*10<sup>-3</sup> milliseconds, where interval is the minimum operator timeout interval supported by hardware (obtainable through the aclrtGetOpTimeoutInterval API), and N is an integer ranging from 1 to 254. If the configured timeout duration is not equal to interval\*N\*10<sup>-3</sup> milliseconds, the latter is used.

<!-- npu="910" id1 -->

- **For Atlas training products**: The unit is second. The value range is \(0, 17340\], and the default value is `1836`. Integer-second precision is allowed.

    Note: For Atlas training products, the actual timeout duration set by the system equals the environment variable value first divided by 68, rounded with the decimal places removed, and then multiplied by 68, in seconds. If the value is less than 68, the system uses a 68s timeout by default.

    For example, if HCCL_EXEC_TIMEOUT=600, the actual timeout duration set by the system is 544s (First, divide 600 by 68. Next, remove the decimal places and get 8. Finally, multiply 8 by 68, and you get 544.)

<!-- end id1 -->

<!-- npu="310p" id2 -->

- **For Atlas inference products**: The unit is second. The value range is \(0, 17340\], and the default value is `1836`. Integer-second precision is allowed.

    Note: For Atlas inference products, the actual timeout duration set by the system equals the environment variable value first divided by 68, rounded with the decimal places removed, and then multiplied by 68, in seconds. If the value is less than 68, the system uses a 68s timeout by default.

    For example, if HCCL_EXEC_TIMEOUT=600, the actual timeout duration set by the system is 544s (First, divide 600 by 68. Next, remove the decimal places and get 8. Finally, multiply 8 by 68, and you get 544.)

<!-- end id2 -->

> [!NOTE] Note
> Generally, you can keep the default value. If the default value cannot meet the requirements for communication synchronization between devices, you can use this environment variable to appropriately increase the synchronization wait time between devices.

## Configuration Example

```bash
export HCCL_EXEC_TIMEOUT=1800
```

## Constraints

If you call the HCCL C API to initialize a communicator with specific configurations and configure the synchronization wait time for inter-device execution through `hcclExecTimeOut` of `HcclCommConfig`, the communicator-level configuration takes precedence.

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products

<!-- npu="910" id3 -->

Atlas training products

<!-- end id3 -->

<!-- npu="310p" id4 -->

Atlas inference products

<!-- end id4 -->