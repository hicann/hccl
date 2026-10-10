# HCCL_UB_MULTI_CHANNEL_NUM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:26:04.464Z pushedAt=2026-09-18T08:03:58.244Z -->

## Description

When UB protocol (CTP) and CLOS topology are used for communication, you can use this environment variable to set the number of channels established for the same EID pair.

This environment variable must be set to an integer. Value range: \[1, 16\]. Default: 1.

Each channel is mapped to an independent jetty at the underlying layer. Multiple channels transfer data in parallel to improve communication bandwidth (similar to the RoCE multi-QP mechanism). The default value is `1`, indicating that the multi-channel feature is disabled.

## Configuration Example

```bash
export HCCL_UB_MULTI_CHANNEL_NUM=4
```

## Constraints

- This environment variable takes effect only when the pure CLOS topology (Level0Shape::CLOS) and UB protocol (CTP) are used and the algorithm runs on the AICPU engine. It does not take effect in links when they use a mesh topology, ubmem protocol, or PCIE. (On the CCU engine, multiple channels share the same jetty, so this environment variable does not take effect.)
- This environment variable supports only collective communication operators (the communicator configuration is read through `HcclConfigGetInfo`). It does not support use cases where self-written test cases directly create channels.
- If the value is beyond \[1, 16\], an error is reported during HCCL initialization or reading. (Error code EI0001 is reported during parsing, and HCCL_E_PARA is returned during reading.)
- Each channel occupies hardware resources such as jetties. Setting too many channels may preempt the resources of other communicators. Set this variable properly (for example, `2` or `4`).
- Note: Increasing the number of channels is beneficial only when a single jetty is the communication bandwidth bottleneck. If a single jetty has reached the link bandwidth limit, multiple channels provide no gain. Set this environment variable based on the performance verification result.

## Applicable Products

<!-- npu="950" id1 -->
- 950PR/950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
