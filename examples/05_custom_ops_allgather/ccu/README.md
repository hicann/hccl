# 自定义通信算子 - AllGather

## 样例介绍

本样例展示如何基于 HCCL 通信编程接口开发 AllGather 通信算子，包含以下功能点：

1. 基于 CCU_SCHED 通信引擎实现 AllGather 集合通信算子
2. 支持自定义算子包的独立构建、独立部署

## 目录结构

```text
├── CMakeLists.txt                      # 编译/构建配置文件
├── op_host/
│   ├── allgather.cc                    # HcclAllGatherCustom 算子实现源文件
│   ├── utils.cc                        # 工具模块（通道获取、线程获取、Kernel注册）
│   └── utils.h                         # 工具模块头文件
├── op_kernel_ccu/
│   ├── ccu_kernel.cc                   # CCU Kernel 实现逻辑
│   ├── ccu_kernel.h                    # CCU Kernel 头文件
│   ├── exec_op.cc                      # CCU 算子编排逻辑
│   └── exec_op.h                       # CCU 算子编排头文件
├── inc/
│   ├── hccl_custom_allgather.h         # 自定义 allgather 算子接口头文件
│   ├── common.h                        # 公共类型头文件
│   └── log.h                           # 日志宏定义
└── testcase/
    ├── main.cc                         # 样例实现源文件
    └── Makefile                        # 编译/构建配置文件
```

> 自定义算子编译工程依赖 HCCL 代码仓中的 [cmake](../../../cmake) 配置和编译脚本 [build.sh](../../../build.sh)，其中：
> 
> - cmake 包含 CMake 配置、MakeSelf 打包配置等内容
> - build.sh 是工程编译入口

## 一、环境准备

### 1. 环境要求

本样例支持以下昇腾产品，组网为单机N卡（N>=2）：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

本样例编译用到的软件依赖如下，注意满足版本号要求：

- gcc & g++ >= 7.3.0
- cmake >= 3.16.0

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装与样例配套的 9.1.0 版本 CANN Toolkit 开发套件包。

安装方式支持默认路径安装和指定路径安装两种方式：

- **默认路径安装**：无需指定 `--install-path` 参数，安装完成后软件位于 `/usr/local/Ascend`（root 用户）或 `${HOME}/Ascend`（非 root 用户）。
- **指定路径安装**：通过 `--install-path` 参数指定安装目录，安装完成后软件位于 `${install_path}`，后续编译及环境变量配置均需使用该路径。

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
# 默认路径安装
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install
# 指定路径安装（${install_path} 替换为实际安装目录）
# ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
```

### 3. 配置环境变量

安装完 CANN Toolkit 开发套件包后，需根据安装方式选择对应的环境变量配置命令，使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装，${install_path}表示CANN-Toolkit包实际安装路径
# source ${install_path}/cann/set_env.sh
```

set_env.sh是环境初始化脚本，负责配置PATH、LD_LIBRARY_PATH、PYTHONPATH等环境变量，使编译器和运行时能正确找到昇腾 NPU 的工具链、库文件和算子库。

## 二、编译自定义算子包

hccl代码仓提供了自定义算子编译打包工程，该工程依赖代码仓中的如下文件：

```text
├── build.sh                        # hccl代码仓根目录编译工程入口
├── CMakeLists.txt                  # hccl代码仓根目录编译/构建配置文件
├── cmake/
│   ├── config.cmake                # CMake变量定义
│   ├── func.cmake                  # CMake函数定义
│   ├── package.cmake               # 签名、打包函数定义
│   └── makeself_custom.cmake       # MakeSelf打包逻辑
└── scripts/
    ├── custom/install.sh           # 自定义算子包安装脚本
    └── sign/add_header_sign.py     # 算子包签名脚本
```

因此，开发者首先需要下载hccl代码仓，然后在代码仓根目录下，执行 `build.sh` 进行编译，通过 `custom_ops_path` 指定自定义算子工程路径：

```bash
# 下载hccl代码仓
git clone https://gitcode.com/cann/hccl.git

# 编译自定义算子包
cd hccl
bash build.sh --vendor=cust --ops=allgather_ccu --custom_ops_path=./examples/05_custom_ops_allgather/ccu
```

> 其中：
> 
> - `--vendor` 参数表示自定义算子标识
> - `--ops` 参数表示自定义算子名称
> - `--custom_ops_path` 参数表示自定义算子工程路径

## 三、安装自定义算子包

自定义算子安装包在 `./build_out` 目录下，通过 `--install` 参数进行安装：

```bash
./build_out/cann-hccl_custom_allgather_ccu_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> 其中：
> 
> - `<arch>` 是当前编译环境的系统架构
> - `<ascend_cann_path>` 是可选参数，表示 CANN 软件包安装目录。默认为 `ASCEND_CUSTOM_OPP_PATH` 或 `ASCEND_OPP_PATH` 环境变量所在的CANN软件包路径

自定义算子包安装信息如下：

- 头文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_allgather.h`
- 动态库：`${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_allgather.so`

> 其中：`${ASCEND_HOME_PATH}`为CANN-Toolkit安装路径

## 四、执行自定义算子

### 1. 编译测试样例

在 `examples/05_custom_ops_allgather/ccu/testcase` 代码目录下执行如下命令：

```bash
# 编译测试样例
make
```

### 2. 执行测试样例

在 `examples/05_custom_ops_allgather/ccu/testcase` 代码目录下执行如下命令：
 	 
```bash
# 运行测试样例
make test

# 或直接执行样例二进制
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH}
./custom_allgather_ccu
```

### 3. 样例结果示例

所有节点的输入数据初始化为该节点的 DeviceId。运行成功后，终端将输出类似以下的日志信息（以 2 卡运行为例）：

```text
Found 2 NPU device(s) available
rankId: 1, input: [ 1 ]
rankId: 0, input: [ 0 ]
rankId: 0, output: [ 0 1 ]
rankId: 1, output: [ 0 1 ]
```
