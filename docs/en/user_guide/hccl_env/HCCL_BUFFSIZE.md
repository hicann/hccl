# HCCL_BUFFSIZE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:06.603Z pushedAt=2026-08-11T03:21:34.392Z -->

## Function

This environment variable is used to control the size of the shared data buffer used by a communicator. It must be set to an integer greater than or equal to 1. The default value is 200, in MB.

In collective communication, each communicator occupies a buffer of HCCL_BUFFSIZE. If there are many communicators in the cluster, the total buffer use increases, which may affect the normal storage of model data. In such cases, you can reduce the value of this environment variable to lower the buffer space occupied by communicators. If your service has a small model data volume but a large communication data volume, you can appropriately increase the value of this environment variable to enlarge the buffer space occupied by communicators, thereby improving data communication efficiency.

The recommended config for LLMs is:

\(MicrobatchSize \* SequenceLength \* hiddenSize \* sizeof \(DataType\) \)/\(1024\*1024\), rounded up.

This environment variable is typically used in the following cases:

- Networks with dynamic shapes.

- Developers call the HCCL C APIs for framework integration.

Note the following:

- The memory requested by this environment variable is exclusive to HCCL and cannot be reused with other service memory.

- Each communicator occupies memory of size `2 x HCCL_BUFFSIZE`, used for sending and receiving respectively.

- This resource is managed at the communicator granularity. Each communicator exclusively owns a set of memory of size `2 x HCCL_BUFFSIZE`, ensuring that concurrent operators across multiple communicators do not interfere with each other.

- For collective communication operators, performance may degrade when the data volume exceeds the value of `HCCL_BUFFSIZE`. It is recommended that the value of `HCCL_BUFFSIZE` be greater than the data volume.

## Configuration Example

```bash
export HCCL_BUFFSIZE=200
```

## Constraints

If you call the HCCL C API to initialize a communicator with specific configurations and configure the shared data buffer size through `hcclBufferSize` of `HcclCommConfig`, the communicator-level configurations take precedence.

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