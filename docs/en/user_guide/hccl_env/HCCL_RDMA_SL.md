# HCCL_RDMA_SL

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:57:00.173Z pushedAt=2026-08-10T08:15:47.271Z -->

## Function

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

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products

<!-- npu="910" id1 -->

Atlas training products

<!-- end id1 -->

<!-- npu="310p" id2 -->

Atlas inference products

<!-- end id2 -->