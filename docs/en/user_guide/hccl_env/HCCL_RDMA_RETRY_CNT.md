# HCCL_RDMA_RETRY_CNT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:12:48.881Z pushedAt=2026-09-18T08:03:58.230Z -->

## Description

Configures the retry count for RDMA NICs. The value must be an integer ranging from 1 to 7, and defaults to 7.

## Configuration Example

```bash
# Set the retry count to 5.
export HCCL_RDMA_RETRY_CNT=5
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id3 -->
- 950PR/950DT: Supported
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas inference products: Supported
<!-- end id2 -->
