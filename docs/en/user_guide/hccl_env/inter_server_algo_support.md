# Inter-Server Communication Algorithm Support List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:09:55.619Z pushedAt=2026-10-15T08:46:37.292Z -->

The following lists the algorithms supported by different inter-server product models, along with the supported communication operators under each algorithm. Those not listed in the tables are not supported.

<!-- npu="950" id2 -->
## Ascend 950PR&950DT Products

- **NHR**

  | Operator | Data Type | Networking Mode | Operator Expansion Mode |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | AllGather | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfp16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | AllReduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Broadcast | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfp16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Reduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Scatter | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfp16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator | AI_CPU/CCU_SCHED |
<!-- end id2 -->

<!-- npu="A3" id3 -->
## Atlas A3 Products

- **Ring**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | ReduceScatterV | int8, int16, int32, int64 (this data type is supported only in single operator mode), float16, float32, bfp16 | - Single-operator | Auto selects NHR or H-D_R algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR or H-D_R algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR or H-D_R algorithm. |

- **NHR**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, int64 (supported only in single operator mode), float16, float32, bfp16 | - Single-operator | Auto selects H-D_R or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects H-D_R or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects H-D_R or ring algorithm. |

- **NB**
  
  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, int64 (supported only in single operator mode), float16, float32, bfp16 | - Single-operator | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR, H-D_R, or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR, H-D_R, or ring algorithm. |

- **AHC**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
<!-- end id3 -->

<!-- npu="910b" id4 -->
## Atlas A2 Products

- **Ring**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR or H-D_R algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |

- **H-D_R**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |

- **NHR**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects H-D_R or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |

- **NHR_V1**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |

- **NB**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR, H-D_R, or ring algorithm. |

- **Pipeline**

  **NOTE**: For Atlas A2 products, if the pipeline algorithm is selected, deterministic computation cannot be enabled; otherwise, the pipeline algorithm does not take effect.

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AllReduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR): For the overflow mode of floating-point computation, saturation mode is not supported; only INF/NaN mode is supported. | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatter | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AlltoAll | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Dynamic shape scenario in Graph (Ascend IR) | Auto selects pairwise algorithm. |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Dynamic shape scenario in Graph (Ascend IR) | Auto selects pairwise algorithm. |
  | AlltoAllVC | int8, int16, int32, int64, float16, float32, bfp16 | - Dynamic shape scenario in Graph (Ascend IR) | Auto selects pairwise algorithm. |

- **Pairwise**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AlltoAll | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None |
  | AlltoAllVC | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None |

<!-- end id4 -->

<!-- npu="910" id1 -->

## Atlas Training Products

- **Ring**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or H-D_R algorithm. |

- **H-D_R**
  
  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR or ring algorithm. |

- **NHR**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects H-D_R or ring algorithm. |

- **NHR_V1**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |

- **NB**

  | Operator | Data Type | Networking Mode | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto selects NHR, H-D_R, or ring algorithm. |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator | Auto selects NHR, H-D_R, or ring algorithm. |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR, H-D_R, or ring algorithm. |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto selects NHR, H-D_R, or ring algorithm. |
<!-- end id1 -->
