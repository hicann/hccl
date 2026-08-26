# Inter-Server Communication Algorithm Support List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:58:19.218Z pushedAt=2026-08-10T09:28:08.467Z -->

The following lists the algorithms supported by different inter-server product models, along with the supported communication operators under each algorithm, Those not listed in the tables are not supported.

## Ascend 950PR/Ascend 950DT

- **NHR**

  | Operator | Data Type | Network Running | Operator Expansion |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | AllGather | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfp16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | AllReduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Broadcast | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bf16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Reduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | AI_CPU/CCU_SCHED |
  | Scatter | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bf16, fp8-e5m2, fp8-e4m3, hif8, fp8-e8m0 | - Single-operator | AI_CPU/CCU_SCHED |

## Atlas A3 Training Products/Atlas A3 Inference Products

- **Ring**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | ReduceScatterV | int8, int16, int32, int64 (supported only in single operator mode), float16, float32, bfp16 | - Single-operator | Auto switching to NHR or H-D_R algorithm |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR or H-D_R algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR or H-D_R algorithm |

- **NHR**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | ReduceScatterV | int8, int16, int32, int64 (supported only in single operator mode), float16, float32, bfp16 | - Single-operator | Auto switching to H-D_R or ring algorithm |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to H-D_R or ring algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to H-D_R or ring algorithm |

- **NB**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | ReduceScatterV | int8, int16, int32, int64 (supported only in single operator mode), float16, float32, bfp16 | - Single-operator | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR, H-D_R, or ring algorithm |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR, H-D_R, or ring algorithm |

- **AHC**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |

## Atlas A2 Training Products/Atlas A2 Inference Products

- **Ring**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR or H-D_R algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |

- **H-D_R**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |

- **NHR**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to H-D_R or ring algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |

- **NHR_V1**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |

- **NB**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR, H-D_R, or ring algorithm |

- **Pipeline**

  **Note**: For Atlas A2 training products/Atlas A2 inference products, if the pipeline algorithm is selected, deterministic computation is not supported; otherwise, the pipeline algorithm will not take effect.

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AllReduce | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR): For the overflow mode of floating-point computation, saturation mode is not supported; only INF/NaN mode is supported. | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | ReduceScatter | int8, int16, int32, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AlltoAll | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Dynamic shapes in graph mode (Ascend IR) | Auto switching to pairwise algorithm |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Dynamic shapes in graph mode (Ascend IR) | Auto switching to pairwise algorithm |
  | AlltoAllVC | int8, int16, int32, int64, float16, float32, bfp16 | - Dynamic shapes in graph mode (Ascend IR) | Auto switching to pairwise algorithm |

- **Pairwise**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AlltoAll | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None |
  | AlltoAllVC | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | None |

- **CP**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | AlltoAllV | int8, int16, int32, int64, float16, float32, bfp16 | Single-operator | Auto switching to pairwise algorithm |

<!-- npu="910" id1 -->

## Atlas Training Products

- **Ring**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or H-D_R algorithm |

- **H-D_R**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |
  | Reduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR or ring algorithm |

- **NHR**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to H-D_R or ring algorithm |

- **NHR_V1**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |

- **NB**

  | Operator | Data Type | Network Running | Handling for Unsupported Operators |
  | --- | --- | --- | --- |
  | ReduceScatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGather | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllReduce | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | Broadcast | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator<br>  - Graph (Ascend IR) | Auto switching to NHR, H-D_R, or ring algorithm |
  | ReduceScatterV | int8, int16, int32, float16, float32, bfp16 | - Single-operator | Auto switching to NHR, H-D_R, or ring algorithm |
  | AllGatherV | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR, H-D_R, or ring algorithm |
  | Scatter | int8, int16, int32, int64, float16, float32, bfp16 | - Single-operator | Auto switching to NHR, H-D_R, or ring algorithm |

<!-- end id1 -->