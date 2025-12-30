# HCCL 代码示例

本目录提供了不同场景下使用 HCCL 接口实现集合通信功能的示例代码。

## 点对点通信

- [HcclSend/HcclRecv（基础收发功能）](./01_point_to_point/01_send_recv/)
- [HcclBatchSendRecv（实现 Ring 环状通信）](./01_point_to_point/02_batch_send_recv_ring/)

## 集合通信

- [AllReduce](./02_collectives/01_allreduce/)
- [Broadcast](./02_collectives/02_broadcast/)
- [AllGather](./02_collectives/03_allgather/)
- [ReduceScatter](./02_collectives/04_reduce_scatter/)
- [Reduce](./02_collectives/05_reduce/)
- [AlltoAll](./02_collectives/06_alltoall/)
- [AlltoAllV](./02_collectives/07_alltoallv/)
- [AlltoAllVC](./02_collectives/08_alltoallvc/)
- [Scatter](./02_collectives/09_scatter/)

## AI 框架

- [PyTorch](./03_ai_framework/01_pytorch/)
- [Tensorflow](./03_ai_framework/02_tensorflow/)

## 自定义点对点通信算子

- [自定义 Send/Recv 算子（基于 AICPU 通信引擎）](./04_custom_ops_p2p/)

## 依赖安装
mpich下载地址，选择4.1.3版本
```bash
https://www.mpich.org/static/downloads/4.1.3/mpich-4.1.3.tar.gz
```

### 编译安装

```shell
tar -zxvf mpich-4.1.3.tar.gz
cd mpich-4.1.3.tar.gz

./configure --disable-fortran  --prefix=/usr/local/mpich --with-device=ch3:nemesis
--prefix=可以指定安装路径

make -j32 && make install
```

## 用例执行
1、设置MPI安装路径环境变量

```shell
export MPI_HOME=/usr/local/mpich
```

2、设置环境变量
```shell
source /usr/local/Ascend/cann/set_env.sh
该脚本和装包路径有关
```
3、执行前冒烟用例
```shell
bash build.sh --cb_test_verify
```
