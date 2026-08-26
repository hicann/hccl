# HCCL_DETERMINISTIC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:53.790Z pushedAt=2026-08-06T07:48:36.320Z -->

## Function

This environment variable configures whether to enable deterministic computation or order preservation for reduction communication operators. Reduction communication operators include AllReduce, ReduceScatter, ReduceScatterV, and Reduce. Reduction order preservation refers to strict deterministic computation, which ensures consistent reduction order on top of determinism.

After deterministic computation or order preservation is enabled for reduction operators, the operators will produce the same output across multiple executions under the same hardware and input conditions.

HCCL_DETERMINISTIC supports the following values:

- `false` (default): Disables deterministic computation.

  - For Atlas A2 training/inference products, deterministic computation is disabled by default for all reduction operators.

  - For Atlas A3 training/inference products:

    - If the expansion mode of the reduction operators is AI CPU: All reduction operators are forced to use deterministic computation, regardless of this configuration.

    - If the expansion mode of the reduction operators is Vector Core: Only AllReduce and ReduceScatter involve non-deterministic computation, which is disabled by default.

  - Ascend 950PR/950DT: All reduction operators are forced to use deterministic computation, regardless of this configuration.

- `true`: Enables deterministic computation for reduction operators.

  - For Atlas A2 training/inference products: AllReduce, ReduceScatter, ReduceScatterV, and Reduce operators are supported.

  - For Atlas A3 training/inference products:

    - If the expansion mode of the reduction operators is AI CPU: All reduction operators are forced to use deterministic computation and are not affected by this configuration.

    - Takes effect on AllReduce and ReduceScatter only when the expansion mode of the reduction operators is Vector Core.

  - For Ascend 950PR/950DT: All reduction operators are forced to use deterministic computation and are not affected by this configuration.

- `strict`: Enables strict deterministic computation for reduction operators, that is, enabling order preservation (the reduction order of all bits is consistent on the basis of determinism). When this parameter is configured, the following conditions must be met:

  - Run only in INF/NaN mode, not the saturation mode.

  - Compared with deterministic computation, enabling order preservation causes a certain degree of performance degradation but is recommended for inference tasks.

  - For Atlas A2 training products/Atlas A2 inference products:

    - Only multi-node symmetric distribution is supported. Asymmetric distribution (i.e., asymmetric device count) is not supported.

    - Reduction operators AllReduce, ReduceScatter, and ReduceScatterV are supported.

  - For Atlas A3 training products and Atlas A3 inference products:

    - Only multi-node symmetric distribution is supported. Asymmetric distribution (i.e., asymmetric device count) is not supported.

    - Reduction operators AllReduce and ReduceScatter are supported. Data types float16, float32, and bfp16 are supported. Only the sum reduction operation is supported.

    - The communication scale requires rank size ≥ 3.

    - If multiple AI servers exist within a SuperPoD, they can communicate only via SDMA communication using HCCS links, not RDMA communication using RoCE. This means that you can't set the environment variable [HCCL_INTER_HCCS_DISABLE](HCCL_INTER_HCCS_DISABLE.md) to "TRUE".

  - For Ascend 950PR/Ascend 950DT:

    - Reduction operators AllReduce and ReduceScatter are supported.

    - The communication scale requires rank size ≥ 3.

    - Only the AI_CPU expansion mode is supported for reduction operators. When order preservation is enabled for other expansion modes (CCU_MS, CCU_SCHED, AIV), they fall back to AI_CPU to enable order preservation.

In general, you don't need to enable deterministic computation for reduction operators. When model execution results vary across multiple runs or during precision tuning, you can use this environment variable to enable deterministic computation for auxiliary debugging and tuning. However, once enabled, the operator execution slows, leading to performance degradation.

If deterministic computation is enabled through this environment variable while the operator expansion mode is set to `AIV` (see [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)), deterministic computation takes precedence, and the `AIV` expansion mode may not take effect in certain cases.

## Configuration Example

```bash
export HCCL_DETERMINISTIC=true
```

## Constraints

If you call the HCCL C API to initialize a communicator with specific configurations and configure the deterministic computation switch through `hcclDeterministic` of `HcclCommConfig`, the communicator-level configuration takes precedence.

## Applicable Products

Atlas A2 training products/Atlas A2 inference products

Atlas A3 training products/Atlas A3 inference products

Ascend 950PR/Ascend 950DT
