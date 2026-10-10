# HCCL_RDMA_SL

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:14:19.337Z pushedAt=2026-09-18T08:03:58.233Z -->

## Description

Configures the service level for RDMA NICs. This value must be consistent with the PFC priority configured on the NIC. Inconsistent configuration may cause performance degradation.

This environment variable must be set to an integer ranging from 0 to 7 and defaults to 4.

## Configuration Example

```bash
# Set the priority to 3.
export HCCL_RDMA_SL=3
```

## Constraints

If you call the HCCL C API to initialize a communicator with specific configurations, and the RDMA NIC service level is configured through `hcclRdmaServiceLevel` of `HcclCommConfig`, the communicator-level configuration takes precedence.

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
