# 自定义通信算子 - AllGather

## 样例介绍

本样例展示如何基于 HCCL 通信编程接口开发 AllGather 通信算子，包含以下功能点：

1. 基于 AICPU 通信引擎实现 AllGather 集合通信算子
2. 支持自定义算子包的独立构建、独立部署

## 目录结构

```text
├── CMakeLists.txt                      # 编译/构建配置文件
├── op_host/
│   ├── allgather.cc                    # HcclAllGatherCustom 算子实现源文件
│   ├── load_kernel.cc                  # AICPU Kernel 在 Host 侧的加载逻辑
│   ├── launch_kernel.cc                # AICPU Kernel 在 Host 侧的下发逻辑
│   ├── launch_kernel_asc.asc           # ASC 语法 Kernel 下发封装
│   └── utils.cc                        # 工具模块
├── op_kernel_aicpu/
│   ├── libcustom_allgather_aicpu_kernel.json  # AICPU Kernel 算子描述文件
│   ├── aicpu_kernel.cc                 # AICPU Kernel 实现逻辑（C++模式）
│   ├── aicpu_kernel_asc.aicpu          # AICPU Kernel 实现逻辑（ASC模式）
│   └── exec_op.cc                      # AICPU 算子编排逻辑
├── inc/
│   ├── hccl_custom_allgather.h         # 自定义 allgather 算子接口头文件
│   ├── common.h                        # 公共类型头文件
│   └── log.h                           # 日志宏定义
├── scripts/
│   └── hccl_custom_allgather_aicpu_check_cfg.xml  # 签名配置文件
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

本样例支持以下昇腾产品, 组网为单机N卡（N>=2）：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

本样例编译用到的软件依赖如下，注意满足版本号要求：

- gcc & g++ : 7.3.0 至 13.3.x
- cmake >= 3.16.0

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装，${install_path}表示CANN-Toolkit包实际安装路径
source ${install_path}/cann/set_env.sh
```

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
    └── sign/add_header_sign.py     # AICPU 算子包签名脚本
```

因此，开发者首先需要下载hccl代码仓，然后在代码仓根目录下，执行 `build.sh` 进行编译，通过 `custom_ops_path` 指定自定义算子工程路径：

```bash
# 下载hccl代码仓
git clone https://gitcode.com/cann/hccl.git

# 编译自定义算子包
cd hccl
bash build.sh --vendor=cust --ops=allgather_aicpu --custom_ops_path=./examples/05_custom_ops_allgather/aicpu
```

> 其中：
> 
> - `--vendor` 参数表示自定义算子标识
> - `--ops` 参数表示自定义算子名称
> - `--custom_ops_path` 参数表示自定义算子工程路径

## 三、安装自定义算子包

自定义算子安装包在 `./build_out` 目录下，通过 `--install` 参数进行安装：

```bash
./build_out/cann-hccl_custom_allgather_aicpu_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> 其中：
> 
> - `<arch>` 是当前编译环境的系统架构
> - `<ascend_cann_path>` 是可选参数，表示 CANN 软件包安装目录。默认为 `ASCEND_CUSTOM_OPP_PATH` 或 `ASCEND_OPP_PATH` 环境变量所在的CANN软件包路径

自定义算子包安装信息如下：

- 头文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_allgather.h`
- 动态库：`${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_allgather_aicpu.so`
- AICPU 算子描述文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/libcustom_allgather_aicpu_kernel.json`
- AICPU 算子包：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_allgather.tar.gz`
- 安装脚本：`${ASCEND_HOME_PATH}/opp/vendors/cust/scripts/install.sh`

> 其中：`${ASCEND_HOME_PATH}`为CANN-Toolkit安装路径

## 四、执行自定义算子

### 1. 关闭 AICPU 算子验签功能

源码编译生成的AICPU算子包`aicpu_hccl_custom_allgather.tar.gz`会在业务启动时加载至Device，加载过程中默认会由驱动进行安全验签，确保包可信。由于开发者通过源码自行编译生成的算子包不含签名信息，所以需要关闭驱动安全验签的机制。

**关闭验签方式：**

    配套使用Ascend HDK 25.5.T2.B001及以上版本，并通过该Ascend HDK自带的npu-smi工具关闭验签。以下为参考命令，需要以root用户在物理机上执行（以device 0为例）。

    ```shell
    npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # 开启验签配置
    npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # 关闭用户自定义验签
    ```

    其中：

    - `-i` 用于指定设备ID，即通过"npu-smi info -l"命令查出的NPU ID
    - `-d` 用于指定对应配置项的属性值

    > 注意：
    > 关闭驱动安全验签机制存在一定的安全风险，需要用户自行确保自定义通信算子的安全可靠，防止恶意攻击行为。

### 2. 修改 AICPU 白名单

AICPU 默认只加载白名单中配置的包，用户自行开发的 AICPU 算子包需配置到白名单中：

```bash
# 编译文件，以 root 用户默认安装路径为例
vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
```

将下列内容追加到 `ascend_package_load.ini` 中：

```ini
name:aicpu_hccl_custom_allgather.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/aicpu/kernel
load_as_per_soc:false
```

各字段含义如下：

- `name`: tar 包文件名
- `install_path`: Device侧的安装路径枚举值，AICPU kernel文件路径须设置为2
- `optional`: 默认为 true，若对应的包不存在，则跳过加载
- `package_path`: tar 包在Host侧CANN Toolkit包下的相对路径
- `load_as_per_soc`: 是否每种芯片类型都加载

### 3. 编译样例

在 `examples/05_custom_ops_allgather/aicpu/testcase` 代码目录下执行如下命令：

```bash
# 编译样例
make
```

### 4. 执行样例

在 `examples/05_custom_ops_allgather/aicpu/testcase` 代码目录下执行如下命令：

```bash
# 运行样例
make test

# 或直接执行样例二进制
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH}
./custom_allgather_aicpu
```

### 5. 样例结果示例

所有节点的输入数据初始化为该节点的 DeviceId。运行成功后，终端将输出类似以下的日志信息（以 2 卡运行为例）：

```text
Found 2 NPU device(s) available
rankId：1, input: [1 1]
rankId: 0, input: [0 0]
rankId: 0, output: [ 0 0 1 1 ]
rankId: 1, output: [ 0 0 1 1 ]
```
