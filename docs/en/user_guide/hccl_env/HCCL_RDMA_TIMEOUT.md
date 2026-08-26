# HCCL_RDMA_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:57:31.987Z pushedAt=2026-08-10T08:35:28.196Z -->

## Function

Configures the coefficient timeout for the RDMA NIC retry timeout.

The formula for the minimum RDMA NIC retry timeout is: *4.096 μs x 2^timeout*, where `timeout` is the configured value of this environment variable, and the actual retry timeout depends on your network conditions.

- For Ascend 950PR/Ascend 950DT:

  - For Atlas 350 accelerator cards, when using a custom RDMA NIC, this environment variable is an integer that ranges from 0 to 31 and defaults to 20. Setting it to `0` or >=32 indicates no timeout.

    > The value of this environment variable is the exponential value of the NACK retry interval for verbs, consistent with the algorithm defined by the verbs API. It is configured by you based on the specifications of the selected RDMA NIC.

- For Atlas A3 training products/Atlas A3 inference products, this environment variable is an integer. Value range: [5, 20]. Default value: 20.

- For Atlas A2 training products/Atlas A2 inference products, this environment variable is an integer. Value range: [5, 20]. Default value: 20.

<!-- npu="910" id1 -->

- For Atlas training products, this environment variable is an integer. Value range: [5, 24]. Default value: 20.<!-- end id1 -->

<!-- npu="310p" id2 -->

- For Atlas inference products, this environment variable is an integer. Value range: [5, 24]. Default value: 20.<!-- end id2 -->

## Configuration Example

```bash
# Set the RDMA NIC retry timeout coefficient to 6. When RDMA is enabled on the NIC, the minimum retry timeout is: 4.096 μs x 2^6
export HCCL_RDMA_TIMEOUT=6
```

## Constraints

None.

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products

<!-- npu="910" id3 -->

Atlas training products

<!-- end id3 -->

<!-- npu="310p" id4 -->

Atlas inference products

<!-- end id4 -->