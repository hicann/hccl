# HCCL_RDMA_RETRY_CNT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:56:53.491Z pushedAt=2026-08-10T08:53:21.753Z -->

## Function

Configures the retry count for RDMA NICs. The value must be an integer ranging from 1 to 7, and defaults to 7.

## Configuration Example

```bash
# Set the retry count to 5.
export HCCL_RDMA_RETRY_CNT=5
```

## Constraints

None.

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products

<!-- npu="910" id1 -->

Atlas training products

<!-- end id1 -->

<!-- npu="310p" id2 -->

Atlas inference products

<!-- end id2 -->