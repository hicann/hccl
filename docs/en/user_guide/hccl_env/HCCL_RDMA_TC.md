# HCCL_RDMA_TC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:57:25.810Z pushedAt=2026-08-10T08:19:07.255Z -->

## Function

Configures the traffic class for RDMA NICs.

The value must be an integer multiple of 4, which ranges from 0 to 255 and defaults to 132.

In the RoCE V2 protocol, this value corresponds to the ToS (Type of Service) field in the IP packet header. The field has 8 bits, where bit[0,1] is fixed at 0 and bits 2-7 are DSCP. Therefore, dividing this value by 4 yields the DSCP value.

![](figures/tos.png)

## Configuration Example

```bash
# // Set this environment variable to 100 (25 x 4), so DSCP is 25.
export HCCL_RDMA_TC=100
```

## Constraints

If you call the HCCL C API to initialize a communicator with specific configurations and configure the RDMA NIC traffic class through `hcclRdmaTrafficClass` of `HcclCommConfig`, the communicator-level configuration takes precedence.

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