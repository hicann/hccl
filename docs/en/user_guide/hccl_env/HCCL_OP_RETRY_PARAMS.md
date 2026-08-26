# HCCL_OP_RETRY_PARAMS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:56:28.878Z pushedAt=2026-08-10T07:44:10.582Z -->

## Function

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

Atlas A3 training products/Atlas A3 inference products
