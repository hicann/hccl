# Inter-Server Communication Algorithm Support List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:46:34.845Z pushedAt=2026-09-18T08:03:58.261Z -->

The following lists the algorithms supported by different inter-server product models, along with the supported communication operators under each algorithm. Those not listed in the tables are not supported.

<!-- npu="950" id2 -->
## 950PR/950DT

- **NHR**

  | Operator | Data Type | Network Mode | Operator Expansion Mode |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | AllGather | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfp16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | AllReduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Broadcast | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfp16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Reduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Scatter | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfp16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator | AI_CPU/CCU_SCHED |
<!-- end id2 -->

<!-- npu="A3" id3 -->
## Atlas A3 Training Products/Atlas A3 Inference Products

- **Ring**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | ReduceScatterV | int8, int16, int32, int64 (this data type is supported only in Single-operator), float16, float32, bfp16 | - Single-operator | Automatically selects NHR or H-D_R algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR or H-D_R algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR or H-D_R algorithm. |

- **NHR**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, int64 (supported only in Single-operator), float16, float32, bfp16 | - Single-operator | Automatically selects H-D_R or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects H-D_R or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects H-D_R or ring algorithm. |

- **NB**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, int64 (supported only in Single-operator), float16, float32, bfp16 | - Single-operator | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR, H-D_R, or ring algorithm. |

- **AHC**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
<!-- end id3 -->

<!-- npu="910b" id4 -->
## Atlas A2 Training Products/Atlas A2 Inference Products

- **Ring**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR or H-D_R algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |

- **H-D_R**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |

- **NHR**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects H-D_R or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |

- **NHR_V1**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |

- **NB**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR, H-D_R, or ring algorithm. |

- **Pipeline**

  **Note**: For Atlas A2 training products/Atlas A2 inference products, if the pipeline algorithm is selected, deterministic computation is not supported; otherwise, the pipeline algorithm will not take effect.

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AllReduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR): For the overflow mode of floating-point computation, saturation mode is not supported; only INF/NaN mode is supported. | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatter | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AlltoAll | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Dynamic shape use cases in graph mode (Ascend IR) | Automatically selects pairwise algorithm. |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Dynamic shape use cases in graph mode (Ascend IR) | Automatically selects pairwise algorithm. |
  | AlltoAllVC | int8, int16, int32, int64, float16, float32, bfp16 | - Dynamic shape use cases in graph mode (Ascend IR) | Automatically selects pairwise algorithm. |

- **Pairwise**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AlltoAll | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None. |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None. |
  | AlltoAllVC | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None. |

- **CP**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | Single-operator | Automatically selects pairwise algorithm. |
<!-- end id4 -->

<!-- npu="910" id1 -->

## Atlas Training Products

- **Ring**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or H-D_R algorithm. |

- **H-D_R**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR or ring algorithm. |

- **NHR**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects H-D_R or ring algorithm. |

- **NHR_V1**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |

- **NB**

  | Operator | Data Type | Network Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Automatically selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator | Automatically selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR, H-D_R, or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Automatically selects NHR, H-D_R, or ring algorithm. |
<!-- end id1 -->
