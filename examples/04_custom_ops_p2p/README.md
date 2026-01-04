# 自定义通信算子 - 点对点通信

## 样例介绍

本样例展示如何基于 HCCL 通信编程接口开发 Send/Recv 点对点通信算子，包含以下功能点：

1. 基于 AICPU 通信引擎实现点对点通信算子
2. 支持自定义算子包的独立构建、独立部署

## 目录结构

```text
├── CMakeLists.txt                      # 编译/构建配置文件
├── op_host/
│   ├── exec_op.cc                      # 算子编排
│   ├── send.cc                         # HcclSendCustom 算子实现源文件
│   └── recv.cc                         # HcclRecvCustom 算子实现源文件
├── op_kernel_aicpu/
│   ├── libp2p_aicpu_kernel.json        # AICPU Kernel 算子描述文件
│   ├── aicpu_kernel.cc                 # AICPU Kernel 实现逻辑
│   ├── load_kernel.cc                  # AICPU Kernel 加载逻辑
│   └── launch_kernel.cc                # AICPU Kernel 下发逻辑
├── inc/
│   ├── hccl_custom_p2p.h               # 自定义 send/recv 算子接口头文件
│   ├── common.h                        # 公共类型头文件
│   └── log.h                           # 日志宏定义
├── scripts/
│   └── hccl_custom_p2p_check_cfg.xml   # 签名配置文件
└── testcase/
    ├── main.cc                         # 样例实现源文件
    └── Makefile                        # 编译/构建配置文件
```

> 自定义算子编译工程依赖 HCCL 代码仓中的 [cmake](../../cmake) 配置和编译脚本 [build.sh](../../build.sh)，其中：
> 
> - cmake 包含 CMake 配置、MakeSelf 打包配置等内容
> - build.sh 是工程编译入口

## 一、环境准备

### 1. 环境要求

本样例支持以下昇腾产品：

- <term>Atlas A3 训练系列产品</term> / <term>Atlas A3 推理系列产品</term>

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh
```

## 二、编译自定义算子包

在代码仓根目录下，执行 `build.sh` 进行编译，通过 `custom_ops_path` 指定自定义算子工程路径：

```bash
bash build.sh --vendor=cust --ops=p2p --custom_ops_path=./examples/04_custom_ops_p2p
```

## 三、安装自定义算子包

自定义算子安装包在 `./build_out` 目录下：

```bash
./build_out/cann-hccl_custom_p2p_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> 其中：
> 
> - `<arch>` 是当前编译环境的系统架构
> - `<ascend_cann_path>` 是 CANN 软件包安装目录

自定义算子包安装信息如下：

- 头文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_p2p.h`
- 动态库：`${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_p2p.so`
- AICPU 算子描述文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/libp2p_aicpu_kernel.json`
- AICPU 算子包：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_p2p.tar.gz`

## 四、执行自定义算子

### 1. 关闭 AICPU 算子验签功能

```bash
# 查询AI CPU算子用户自定义验签能力使能状态
# 0：关闭用户自定义验签能力
# 1：开启用户自定义验签能力
for i in {0..7}; do npu-smi info -t custom-op-secverify-enable -i $i; done

# 设置AI CPU算子用户自定义验签能力使能状态，使能开关
for i in {0..7}; do npu-smi set -t custom-op-secverify-enable -i $i -d 1; done

# 查询AI CPU算子验签模式
# 0：关闭验证，不验签
# 1：华为证书，使用华为证书验签（默认）
# 2：客户自定义证书
# 3：华为证书、客户自定义证书
# 4：开源社区证书
# 5：华为证书、开源社区证书
# 6：客户自定义证书、开源社区证书
# 7：华为证书、客户自定义证书、开源社区证书
for i in {0..7}; do npu-smi info -t custom-op-secverify-mode -i $i; done

# 设置AI CPU算子验签模式，关闭验签
for i in {0..7}; do npu-smi set -t custom-op-secverify-mode -i $i -d 0; done
```

> 注意：关闭验签后，需重启机器才能生效

### 2. 修改 AICPU 白名单

AICPU 默认只加载白名单中配置的包，用户自行开发的 AICPU 算子包需配置到白名单中：

```bash
# 编译文件，以 root 用户默认安装路径为例
vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
```

将下列内容追加到 `ascend_package_load.ini` 中：

```ini
name:aicpu_hccl_custom_p2p.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/aicpu/kernel
```

各字段含义如下：

- `name`: tar 包文件名
- `install_path`: 安装到 Device 侧的路径
- `optional`: 默认为 true
- `package_path`: tar 包在Host侧CANN Toolkit包下的相对路径

### 3. 编译样例

在 `examples/04_custom_ops_p2p/testcase` 代码目录下执行如下命令：

```bash
# 编译样例
make
```

### 4. 执行样例

```bash
# 运行样例
make test
```

### 5. 样例结果示例

偶数节点的 `sendBuf` 内容初始化为该节点的 DeviceId，然后将数据发送至下一奇数节点，因此各个奇数节点接收到的是上一节点的 DeviceId。

```text
Found 8 NPU device(s) available
rankId: 1, output: [ 0 0 0 0 ]
rankId: 3, output: [ 2 2 2 2 ]
rankId: 5, output: [ 4 4 4 4 ]
rankId: 7, output: [ 6 6 6 6 ]
```
