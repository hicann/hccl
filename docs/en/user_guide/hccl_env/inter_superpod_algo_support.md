# Inter-SuperPoD Communication Algorithm Support List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:52:58.280Z pushedAt=2026-09-18T08:03:58.264Z -->

The inter-SuperPoD communication algorithm support table described in this section applies only to Atlas A3 training products/Atlas A3 inference products.

- **Ring**

  | Operator | Data Type | Network Mode | Deterministic Computation Supported | Handling for Unsupported Operators |
  | --- | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR or H-D_R algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR or H-D_R algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR or H-D_R algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR or H-D_R algorithm. |
  | ReduceScatterV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the NHR or H-D_R algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the NHR or H-D_R algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the NHR or H-D_R algorithm. |

- **H-D_R**

  | Operator | Data Type | Network Mode | Deterministic Computation Supported | Handling for Unsupported Operators |
  | --- | --- | --- | --- | --- |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR or ring algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR or ring algorithm. |

- **NHR**

  | Operator | Data Type | Network Mode | Deterministic Computation Supported | Handling for Unsupported Operators |
  | --- | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the H-D_R or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the H-D_R or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the H-D_R or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the H-D_R or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the H-D_R or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the H-D_R or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the H-D_R or ring algorithm. |

- **NB**

  | Operator | Data Type | Network Mode | Deterministic Computation Supported | Handling for Unsupported Operators |
  | --- | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |

- **Pipeline**

  | Operator | Data Type | Network Mode | Deterministic Computation Supported | Handling for Unsupported Operators |
  | --- | --- | --- | --- | --- |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator (takes effect only when zero-copy is enabled) | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
  | ReduceScatter | int8, int16, int32, float16, float32, bfp16 | - Single-operator (takes effect only when zero-copy is enabled) | Yes | Automatically selects the NHR, H-D_R, or ring algorithm. |
