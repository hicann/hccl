# AlltoAll  集合通信算子

## 介绍

本工程为 AlltoAll 集合通信算子的赛事开发样例，基于 AICPU 通信引擎实现，分为 Host 侧与 Device 侧两部分：

- `op_host/`：编译生成 `libhccl.so`，提供算子入口 `HcclAlltoAll` 及 AICPU Kernel 的加载/下发逻辑（`LaunchAICPUKernel`）。Host 侧通过 `libhcomm.so` 提供的弱符号（`HcclCommInitRootInfo`、`HcclGetRootInfo`、`HcclGetRankId`、`HcclChannelAcquire` 等）完成通信域初始化与资源管理，因此 `libhccl.so` 可独立运行，替换标准 HCCL 库中的 `HcclAlltoAll` 实现。
- `op_kernel_aicpu/`：交叉编译生成 aarch64 的 `libhccl_device.so`，提供 AICPU Kernel 入口 `HcclAICPUKernel`，在 Device 侧编排 AlltoAll 通信算法（`ExecOp`），通过 `Hcomm*` 原语（`HcommWriteOnThread`、`HcommLocalCopyOnThread`、`HcommChannelNotify*`）完成数据收发。

> **注意**：本工程需搭配 **hcomm 仓竞赛分支（`competition/campus-2026`）下的 HCCL-VM 虚拟执行环境** 使用。HCCL-VM 位于 hcomm 仓 `test/hccl_vm`，提供无真实昇腾硬件条件下的算子执行与校验能力。请确保 hcomm 仓已切换至 `competition/campus-2026` 分支，并按其指导手册（`README-Competition.md`）完成 HCCL-VM 的安装与编译。

## 编译运行

```bash
source /usr/local/Ascend/cann/set_env.sh

bash build.sh
```

## 对接 HCCL-VM 虚拟执行环境

> 本样例需搭配 **hcomm 仓竞赛分支（`competition/campus-2026`）** 的 [HCCL-VM](../../hcomm/test/hccl_vm/README-Competition.md) 工具使用。HCCL-VM 位于 hcomm 仓 `test/hccl_vm`，是无真实昇腾硬件条件下执行/校验 HCCL 算子的虚拟环境。请先切换 hcomm 仓至 `competition/campus-2026` 分支，确保 HCCL-VM 版本与本样例匹配。

本样例采用 AICPU 展开模式，在 HCCL-VM 中的对接流程如下。

### 1. 前置条件

参照 HCCL-VM 指导手册完成工具的安装与编译（一键安装或手动构建）。后续步骤沿用 HCCL-VM 手册的默认路径约定，实际安装路径以本机为准：

| 占位符 | 默认路径 | 说明 |
| --- | --- | --- |
| `<ascend_path>` | `/home/workspace/Ascend` | CANN Toolkit 安装根目录，对应安装时的 `--install-path`；执行 `source <ascend_path>/cann/set_env.sh` 后由 CANN 导出 `ASCEND_HOME_PATH` |
| `<hccl_vm_install>` | `/home/workspace/hcomm/test/hccl_vm/hccl_vm_install` | HCCL-VM 安装目录，在源码目录 `/home/workspace/hcomm/test/hccl_vm` 下执行 `build.sh` 生成；内含 `bin/`（`hccl-vm`）、`lib/aarch64/`（aarch64 设备侧 `.so`）、`data/`（`ranktable.json`、`topo.json`）、`config/`（拓扑配置）等子目录 |

安装与编译完成后，请逐项确认：

- CANN Toolkit 已安装，`source <ascend_path>/cann/set_env.sh` 生效，`$ASCEND_HOME_PATH` 已正确导出。
- HCCL-VM 安装目录 `<hccl_vm_install>` 存在，且 `<hccl_vm_install>/bin/hccl-vm` 可执行。
- **AICPU 展开模式所需的标准 Device 侧符号已通过 `bash build_pkg.sh` 部署**到 `<hccl_vm_install>/lib/aarch64/`（含 `libascend_hal.so`、`libc_sec.so`、`libmmpa.so` 及 aicpu 算子包解压产物）。

> **重要**：`build_pkg.sh` 除了部署设备侧 `.so` 符号外，还会将 HCCL/HCOMM 源码编译安装到 CANN 目录，提供 `hcomm_primitives.h` 等头文件——这是下一步骤 `bash build.sh` 编译样例的前置依赖。若跳过 `build_pkg.sh`，样例编译会报 `fatal error: hcomm_primitives.h: No such file or directory`。

- hccl_test 工具已编译，二进制位于 `<ascend_path>/cann/tools/hccl_test/bin/`（即 `$ASCEND_HOME_PATH/tools/hccl_test/bin/`）。若使用一键安装脚本（`hccl_vm_installer`），hccl_test 默认随之一并编译；手动构建路径下需参照 HCCL-VM 手册「hccl_test 用例构建」章节单独编译。
- `/etc/hccl_rootinfo.json` 已创建并正确指向 `<hccl_vm_install>/data/topo.json`。若该文件不存在，参照 HCCL-VM 手册「hccl_rootinfo.json 文件」章节创建（一键安装路径下通常已自动生成）。

### 2. 构建样例产物

```bash
source <ascend_path>/cann/set_env.sh
bash build.sh
```

构建完成后产物位于 `build/`：

| 产物 | 路径 | 架构 | 说明 |
| --- | --- | --- | --- |
| Host 侧库 | `build/lib64/libhccl.so` | x86_64 | 提供 `HcclAlltoAll`，Host 进程原生加载 |
| Device 侧库 | `build/lib64/libhccl_device.so` | aarch64 | 提供 `HcclAICPUKernel`，由 HCCL-VM 经 QEMU 模拟执行 |
| 头文件 | `build/include/hccl.h` | - | `HcclAlltoAll` 接口声明 |

### 3. 部署 Device 侧符号

HCCL-VM 的 AICPU 模式通过 dlopen 加载 `lib/aarch64/` 下的 aarch64 `.so`，并在 QEMU 中执行其导出符号。将样例 Device 侧库拷贝至该目录：

```bash
cp build/lib64/libhccl_device.so <hccl_vm_install>/lib/aarch64/
chmod 755 <hccl_vm_install>/lib/aarch64/libhccl_device.so
```

> HCCL-VM 不对源码编译的 `.so` 做签名校验，直接拷贝即可（真机环境才需关闭 `npu-smi` 验签并配置白名单）。

### 4. 配置 Host 侧库

样例 `libhccl.so` 提供 `HcclAlltoAll`，并通过 `libhcomm.so` 的弱符号获得通信域初始化等接口。运行用例前，将其置于 `LD_LIBRARY_PATH` 最前，以替换标准 `libhccl.so` 的 `HcclAlltoAll` 实现：

```bash
export LD_LIBRARY_PATH=<样例路径>/build/lib64:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
```

### 5. 配置环境变量并执行

```bash
cd <hccl_vm_install>
source <ascend_path>/cann/set_env.sh
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

cd bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# 选择通信域（示例：1 超节点 1 server 2 卡）
(hvm)$> hccl-vm mock-comm 112

# 运行 alltoall 用例（用例经 LD_LIBRARY_PATH 加载样例 libhccl.so）
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 \
    ${ASCEND_HOME_PATH}/tools/hccl_test/bin/alltoall_test \
    -b 64 -e 64 -d int32 -w 0 -n 1 -c 0 > log.txt

# 执行 Checker 校验 DAG
(hvm)$> hccl-vm plugin run @checker

(hvm)$> exit
```

> hccl_test 各参数含义及支持的取值以 hccl_test 工具说明为准；结果查看参见 HCCL-VM 手册的 [Checker 结果](../../hcomm/test/hccl_vm/README-Competition.md#492-checker插件结果)章节。

### 对接原理

1. 用例调用 `HcclAlltoAll`，命中样例 `libhccl.so` 中的实现（`op_host/alltoall.cc`）。
2. Host 侧经 `libhcomm.so`（由 HCCL-VM 模拟）申请 Thread/Channel 资源，并将 `AlgResourceCtx` 序列化后下发。
3. `LaunchAICPUKernel` 调用 `aclrtBinaryLoadFromFile`（HCCL-VM 桩解析 `aicpu_kernel.json`）与 `aclrtLaunchKernelWithConfig`（HCCL-VM 桩将 `kernelName=HcclAICPUKernel`、`soName=libhccl_device.so` 及参数经管道转发至 Device 进程）。
4. Device 进程（aarch64，QEMU 模拟）在 `lib/aarch64/libhccl_device.so` 中 dlsym `HcclAICPUKernel` 并执行，完成 `ExecOp` 中的 AlltoAll 算法编排。

## 代码格式

```bash
bash build.sh --format
```
