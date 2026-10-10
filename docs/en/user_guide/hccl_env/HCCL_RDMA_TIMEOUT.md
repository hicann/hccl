# HCCL_RDMA_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:17:33.617Z pushedAt=2026-09-18T08:03:58.237Z -->

## Description

Configures the coefficient timeout for the RDMA NIC retry timeout.

The formula for the minimum RDMA NIC retry timeout is: *4.096 μs x 2^timeout*, where `timeout` is the configured value of this environment variable, and the actual retry timeout depends on your network conditions.

<!-- npu="950" id10 -->
- For 950PR/950DT:
  - For Atlas 350 accelerator cards, when using a custom RDMA NIC, this environment variable is an integer that ranges from 0 to 31 and defaults to 20. Setting it to `0` or >=32 indicates no timeout.
    > The value of this environment variable is the exponential value of the NACK retry interval for verbs, consistent with the algorithm defined by the verbs API. It is configured by you based on the specifications of the selected RDMA NIC.
<!-- end id10 -->
<!-- npu="A3" id9 -->
- For Atlas A3 training products/Atlas A3 inference products, this environment variable is an integer. Value range: [5, 20]. Default value: 20.
<!-- end id9 -->
<!-- npu="910b" id8 -->
- For Atlas A2 training products/Atlas A2 inference products, this environment variable is an integer. Value range: [5, 20]. Default value: 20.
<!-- end id8 -->
<!-- npu="910" id1 -->
- For Atlas training products, this environment variable is an integer. Value range: [5, 24]. Default value: 20.
<!-- end id1 -->
<!-- npu="310p" id2 -->
- For Atlas inference products, this environment variable is an integer. Value range: [5, 24]. Default value: 20.
<!-- end id2 -->

## Configuration Example

```bash
# If the coefficient of the RDMA NIC retransmission timeout is configured as 6, the minimum retransmission timeout when the RDMA function is enabled on the NIC is: 4.096μs * 2^6
export HCCL_RDMA_TIMEOUT=6
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id5 -->
- 950PR/950DT: Supported
<!-- end id5 -->
<!-- npu="A3" id6 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id6 -->
<!-- npu="910b" id7 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id7 -->
<!-- npu="910" id3 -->
- Atlas training products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
