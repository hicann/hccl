# HCCL_INTER_HCCS_DISABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:35:51.678Z pushedAt=2026-09-18T08:03:58.201Z -->

## Description

This environment variable is used to configure the communication link type within a SuperPoD in SuperPoD networking. The following values are supported:

- `TRUE`: AI nodes within the SuperPoD use RoCE for RDMA communication.
- `FALSE`: AI nodes within the SuperPoD use HCCS communication links for SDMA communication.

The default value is `FALSE`.

## Configuration Example

```bash
export HCCL_INTER_HCCS_DISABLE=FALSE
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id2 -->
- 950PR/950DT: Not supported
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
