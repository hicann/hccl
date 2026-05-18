# Server间通信算法支持度列表

下面分别给出Server间不同型号的产品支持的算法，以及对应算法下支持的通信算子介绍，表格中未列出的则代表不支持。

## Ascend 950PR/Ascend 950DT

- **NHR算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 通信算子展开模式 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | AI_CPU/CCU_SCHED |
  | AllGather | int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64、bfp16、fp8-e5m2、fp8-e4m3、hif8、fp8-e8m0 | - 单算子模式<br>  - 图模式（Ascend IR） | AI_CPU/CCU_SCHED |
  | AllReduce | int8、int16、int32、float16、float32、 bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | AI_CPU/CCU_SCHED |
  | Broadcast | int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64、bf16、fp8-e5m2、fp8-e4m3、hif8、fp8-e8m0 | - 单算子模式<br>  - 图模式（Ascend IR） | AI_CPU/CCU_SCHED |
  | Reduce | int8、int16、int32、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | AI_CPU/CCU_SCHED |
  | Scatter | int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64、bf16、fp8-e5m2、fp8-e4m3、hif8、fp8-e8m0 | - 单算子模式 | AI_CPU/CCU_SCHED |
  | Broadcast | int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64、bf16、fp8-e5m2、fp8-e4m3、hif8、fp8-e8m0 | - 单算子模式<br>  - 图模式（Ascend IR） | AI_CPU/CCU_SCHED |

## Atlas A3 训练系列产品/Atlas A3 推理系列产品

- **ring算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | Reduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | ReduceScatterV | int8、int16、int32、int64（仅单算子模式支持此数据类型）、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR或者H-D_R算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR或者H-D_R算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR或者H-D_R算法 |

- **NHR算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | ReduceScatterV | int8、int16、int32、int64（仅单算子模式支持）、float16、float32、bfp16 | - 单算子模式 | 自动选择为H-D_R或者ring算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为H-D_R或者ring算法 |

- **NB算法**
  
  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | ReduceScatterV | int8、int16、int32、int64（仅单算子模式支持）、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR、H-D_R或者ring算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR、H-D_R或者ring算法 |

- **AHC算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |

## Atlas A2 训练系列产品/Atlas A2 推理系列产品

- **ring算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | AllGather | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | AllReduce | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | Reduce | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | ReduceScatterV | int8、int16、int32、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | Scatter | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR或者H-D_R算法 |
  | AllGatherV | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |

- **H-D_R算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | Reduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |

- **NHR算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | ReduceScatterV | int8、int16、int32、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |

- **NHR_V1算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |

- **NB算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | ReduceScatterV | int8、int16、int32、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR、H-D_R或者ring算法 |

- **pipeline算法**

  **注意**：针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，如果选择pipeline算法，不支持开启确定性计算；否则pipeline算法不会生效。

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | AllReduce | int8、int16、int32、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR）：针对浮点计算的溢出模式，不支持饱和模式，仅支持INF/NaN模式。 | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | ReduceScatter | int8、int16、int32、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AlltoAll | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR）的动态shape场景 | 自动选择为pairwise算法 |
  | AlltoAllV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR）的动态shape场景 | 自动选择为pairwise算法 |
  | AlltoAllVC | int8、int16、int32、int64、float16、float32、bfp16 | - 图模式（Ascend IR）的动态shape场景 | 自动选择为pairwise算法 |

- **pairwise算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | AlltoAll | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AlltoAllV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AlltoAllVC | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |

- **CP算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | AlltoAllV | int8、int16、int32、int64、float16、float32、bfp16 | 单算子模式 | 自动选择为pairwise算法 |

<cann-filter npu-type="910">
## Atlas 训练系列产品

- **ring算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | AllGather | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | AllReduce | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |
  | Reduce | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者H-D_R算法 |

- **H-D_R算法**
  
  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |
  | Reduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR或者ring算法 |

- **NHR算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为H-D_R或者ring算法 |

- **NHR_V1算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |

- **NB算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 不支持算子处理方法 |
  | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 自动选择为NHR、H-D_R或者ring算法 |
  | ReduceScatterV | int8、int16、int32、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR、H-D_R或者ring算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 自动选择为NHR、H-D_R或者ring算法 |
</cann-filter>
