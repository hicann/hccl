# 超节点间通信算法支持度列表

本节所述的超节点间通信算法支持度表格仅适用于Atlas A3 训练系列产品/Atlas A3 推理系列产品。

- **ring算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 是否支持确定性计算 | 不支持算子处理方法 |
  | --- | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR或者H-D_R算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR或者H-D_R算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR或者H-D_R算法 |
  | Reduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR或者H-D_R算法 |
  | ReduceScatterV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为NHR或者H-D_R算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为NHR或者H-D_R算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为NHR或者H-D_R算法 |

- **H-D_R算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 是否支持确定性计算 | 不支持算子处理方法 |
  | --- | --- | --- | --- | --- |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR或者ring算法 |
  | Reduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR或者ring算法 |

- **NHR算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 是否支持确定性计算 | 不支持算子处理方法 |
  | --- | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为H-D_R或者ring算法 |
  | ReduceScatterV | int8、int16、int32、int64、 float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为H-D_R或者ring算法 |
  | Scatter | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为H-D_R或者ring算法 |

- **NB算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 是否支持确定性计算 | 不支持算子处理方法 |
  | --- | --- | --- | --- | --- |
  | ReduceScatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR、H-D_R或者ring算法 |
  | AllReduce | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR、H-D_R或者ring算法 |
  | Broadcast | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式<br>  - 图模式（Ascend IR） | 是 | 自动选择为NHR、H-D_R或者ring算法 |
  | ReduceScatterV | int8、int16、int32、 int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为NHR、H-D_R或者ring算法 |
  | Scatter | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为NHR、H-D_R或者ring算法 |
  | AllGatherV | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式 | 是 | 自动选择为NHR、H-D_R或者ring算法 |

- **pipeline算法**

  | 集合通信算子 | 数据类型 | 网络运行模式 | 是否支持确定性计算 | 不支持算子处理方法 |
  | --- | --- | --- | --- | --- |
  | AllGather | int8、int16、int32、int64、float16、float32、bfp16 | - 单算子模式（仅开启零拷贝的场景下生效） | 是 | 自动选择为NHR、H-D_R或者ring算法 |
  | ReduceScatter | int8、int16、int32、float16、float32、bfp16 | - 单算子模式（仅开启零拷贝的场景下生效） | 是 | 自动选择为NHR、H-D_R或者ring算法 |
