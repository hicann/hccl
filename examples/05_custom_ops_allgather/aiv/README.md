# 自定义通信算子 - AllGather 通信

## 样例介绍

本样例展示如何基于 HCCL AIV 通信编程接口开发 AllGather 自定义通信算子，主要功能点：

1.  基于 AIV (AI Vector) 通信引擎实现 AllGather 集合通信算子。
2.  包含 Host 侧算子逻辑与 Device 侧 Kernel 实现。
3.  提供完整的编译构建与测试验证流程。

## 目录结构

```text
├── CMakeLists.txt                      # 根目录编译/构建配置文件
├── op_host/
│   ├── CMakeLists.txt
│   ├── all_gather.cc                   # HcclAllGatherCustom 算子Host侧实现
│   ├── launch_kernel.cc                # Kernel 下发逻辑实现
│   └── launch_kernel.h                 # Kernel 下发接口定义
├── op_kernel/
│   ├── CMakeLists.txt
│   └── launch_kernel_asc.asc           # 算子 Kernel 侧实现 (Ascend C)
├── inc/
│   ├── hccl_custom_allgather.h         # 自定义算子对外接口头文件
│   ├── common.h                        # 公共类型定义与宏
│   ├── aiv_all_gather_mesh_1d.h        # AIV AllGather 核心算法实现
│   ├── aiv_communication_base_v2.h     # AIV 通信基类
│   ├── log.h                           # 日志工具
│   ├── extra_args.h                    # 额外参数定义
│   └── sync_interface.h                # 同步接口定义
└── testcase/
    ├── CMakeLists.txt                  # 测试用例 CMake 配置文件
    ├── Makefile                        # 测试用例 Makefile (用于编译运行)
    └── main.cc                         # 测试用例主程序
```

## 一、环境准备

### 1. 环境要求

本样例支持以下产品，组网为单机N卡（N>=2）：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

以 root 用户默认安装路径为例：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

此外，运行测试用例需要 MPI 环境支持，请确保已安装并配置好 MPI。MPI配置请参考配套版本的[昇腾文档中心-HCCL性能测试工具使用指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest)中的“MPI安装与配置”章节。

## 二、编译与运行

本样例提供了基于 CMake 的构建流程以及基于 Makefile 的测试运行脚本。

### 1. 编译自定义算子库

在根目录下执行以下命令：

```bash
bash build.sh --vendor=cust --ops=allgather_aiv --custom_ops_path=./examples/06_custom_ops_allgather/aiv
```
> 其中：
> 
> - `--vendor` 参数表示自定义算子标识
> - `--ops` 参数表示自定义算子名称
> - `--custom_ops_path` 参数表示自定义算子工程路径

### 2. 安装算子包
自定义算子安装包在 `./build_out` 目录下，通过 `--install` 参数进行安装：

```bash
./build_out/cann-hccl_custom_allgather_aiv_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> 其中：
> 
> - `<arch>` 是当前编译环境的系统架构
> - `<ascend_cann_path>` 是可选参数，表示 CANN 软件包安装目录。默认为 `ASCEND_CUSTOM_OPP_PATH` 或 `ASCEND_OPP_PATH` 环境变量所在的CANN软件包路径

### 3. 运行测试用例

测试代码在 `examples/05_custom_ops_allgather/testcase`,在前节`1. 编译自定义算子库`已经编译好测试样例
测试样例二进制文件路径`./build/examples/06_custom_ops_allgather/testcase/custom_allgather_test`

在根目录使用mpirun执行命令
```
mpirun -n rank_size build/examples/06_custom_ops_allgather/testcase/custom_allgather_test data_len
参数说明:
rank_size: 使用的卡数
data_len: 数据长度
```

### 4. 预期结果

运行成功后，终端将输出类似以下的日志信息（以 2 卡运行为例）：

```text
[1786071476.120968] [Rank 0] MPI Initialized. World Size: 2
[1786071476.120968] [Rank 1] MPI Initialized. World Size: 2
[1786071476.127411] [Rank 0] Device 0 selected (Total devices: 8)
[1786071476.127411] [Rank 1] Device 1 selected (Total devices: 8)
[1786071478.023709] [Rank 0] Root info generated
[1786071478.023786] [Rank 0] HCCL set device[0]
[1786071478.023778] [Rank 1] HCCL set device[1]
[1786071483.214938] [Rank 0] HCCL Comm Initialized
[1786071483.221873] [Rank 0] Buffers allocated and initialized
[1786071483.254098] [Rank 1] HCCL Comm Initialized
[1786071483.259378] [Rank 1] Buffers allocated and initialized
rank1 dataLen=1024 time=835 ms
[1786071484.095144] [Rank 1] VerifyResult Passed!
rank0 dataLen=1024 time=873 ms
[1786071484.095200] [Rank 0] VerifyResult Passed!
```
