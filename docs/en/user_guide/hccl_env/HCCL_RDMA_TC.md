# HCCL_RDMA_TC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:16:02.409Z pushedAt=2026-09-18T08:03:58.235Z -->

## Description

Configures the traffic class for RDMA NICs.

The value must be an integer multiple of 4, which ranges from 0 to 255 and defaults to 132.

In the RoCE V2 protocol, this value corresponds to the ToS (Type of Service) field in the IP packet header. The field has 8 bits, where bit[0,1] is fixed at 0 and bits 2-7 are DSCP. Therefore, dividing this value by 4 yields the DSCP value.

![](figures/tos.png)

## Configuration Example

```bash
# If this environment variable is set to 100 (25*4), DSCP is 25.
export HCCL_RDMA_TC=100
```

## Constraints

If you call the HCCL C API to initialize a communicator with specific configurations and configure the RDMA NIC traffic class through `hcclRdmaTrafficClass` of `HcclCommConfig`, the communicator-level configuration takes precedence.

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
