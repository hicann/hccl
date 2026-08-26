# HCCL_MULTI_QP_THRESHOLD

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:55:58.024Z pushedAt=2026-08-06T07:04:57.739Z -->

## Function

When multi-QP communication is used for RDMA communication between ranks, you can use this environment variable to set the threshold of minimum data volume shared by each QP.

This environment variable must be an integer. Range: [1, 8192]. Default: 512. Unit: KB.

- If "(Data volume per communication between ranks/Value of HCCL_RDMA_QPS_PER_CONNECTION) < Value of HCCL_MULTI_QP_THRESHOLD", HCCL automatically reduces the number of QPs during execution so that the data volume shared by each QP is greater than or equal to the value of `HCCL_MULTI_QP_THRESHOLD`. For example:

    The data volume per communication between ranks is 1 MB, HCCL_RDMA_QPS_PER_CONNECTION is set to 4, and HCCL_MULTI_QP_THRESHOLD is set to 512. In this case, each QP is required to share at least 512 KB of data. During HCCL execution, the number of QPs is reduced to 2, and only 2 QPs are used for data transmission between ranks.

- When the inter-rank data volume is less than `HCCL_MULTI_QP_THRESHOLD`, a single QP is used for transmission.

- When each QP handles more than 512 KB of data, RDMA traffic tests using the HCCL Test tool (testing cross-node traffic only, no HCCS links) show that the dispatch and scheduling overhead in multi-QP scenarios incurs less than 3% performance degradation compared to single-QP scenarios.

> [!NOTE] Note
> Multi-QP communication can be enabled through the environment variable [HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md) or [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md).

## Configuration Example

```bash
export HCCL_MULTI_QP_THRESHOLD=512
```

## Constraints

This environment variable supports only single-operator calls and does not support static graphs.

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products
