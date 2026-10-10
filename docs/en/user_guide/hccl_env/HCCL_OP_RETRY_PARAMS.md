# HCCL_OP_RETRY_PARAMS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:05:36.979Z pushedAt=2026-09-18T08:03:58.221Z -->

## Description

When you enable the HCCL operator retry feature through the environment variable [HCCL_OP_RETRY_ENABLE](HCCL_OP_RETRY_ENABLE.md), you can use this environment variable to configure the wait time before the first retry, the maximum number of retries, and the interval between two retries.

The configuration method is as follows:

**export HCCL_OP_RETRY_PARAMS="MaxCnt:3,HoldTime:5000,IntervalTime:1000"**

- `MaxCnt`: Maximum retry count, of uint32 type. Value range: 1-10. Default: 1.
- `HoldTime`: Wait time from detection of communication operator execution failure to the start of the first retry, of uint32 type. Value Range: 0-60000. Default: 5000, in ms.
- `IntervalTime`: Interval between two retries of the same communication operator, of uint32 type. Value range: 0-60000. Default: 1000, in ms.

## Configuration Example

```bash
export HCCL_OP_RETRY_PARAMS="MaxCnt:5,HoldTime:5000,IntervalTime:5000"
```

## Constraints

- This environment variable takes effect only when HCCL retry is enabled through [HCCL_OP_RETRY_ENABLE](HCCL_OP_RETRY_ENABLE.md) (enabling retry at any level suffices).
- If you configure the wait time for the first retry through `hcclRetryParams` of `HcclCommConfig` when calling the HCCL C API to initialize a communicator with specific configurations, the communicator-level configuration takes precedence.

## Applicable Products

<!-- npu="950" id2 -->
- 950PR/950DT: Not Supported
<!-- end id2 -->
<!-- npu="A3" id1 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id1 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
