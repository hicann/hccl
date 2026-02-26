# 源码构建

## 环境准备

1. 安装依赖。

   HCCL集合通信库编译用到的依赖如下，请注意版本要求。

   - python: 3.7.x 至 3.11.4 版本
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - ccache（可选，用于提高二次编译速度）
   - googletest（仅执行UT时依赖，建议版本 release-1.14.0）

2. 安装社区版CANN Toolkit包

   编译本项目依赖CANN Toolkit开发套件包，请根据操作系统架构，从[CANN软件包归档页面](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master/)中下载最新的CANN Toolkit安装包，参考[昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)中的“安装CANN-安装Toolkit开发套件包”章节进行安装：

   ```shell
   # 安装命令，其中--install-path为可选参数，用于指定安装路径
   bash Ascend-cann-toolkit_<version>_linux-<arch>.run --full --install-path=<install_path>
   ```

   - `<cann_version>`: 表示CANN包版本号。
   - `<arch>`: 表示CPU架构，如aarch64、x86_64。
   - `<install_path>`: 表示指定安装路径，可选，root用户默认安装在/usr/local/Ascend目录，指定路径安装时，指定的路径权限需设置为755。

3. 设置CANN软件环境变量。

   ```shell
   # 默认路径，root用户安装
   source /usr/local/Ascend/cann/set_env.sh

   # 默认路径，非root用户安装
   source $HOME/Ascend/cann/set_env.sh
   ```

## 源码下载

```shell
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/hccl.git
```

## 编译

### 开源第三方软件依赖

编译本项目时，依赖的第三方开源软件列表如下：

| 开源软件      | 版本                   | 下载地址                                                                                                                                                                                                    |
| ------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| makeself      | 2.5.0                  | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)                                     |
| googletest    | 1.14.0                 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz)                                                                          |

### 源码编译

本项目提供一键式编译构建能力，进入代码仓根目录，执行如下命令：

```shell
# 编译 host 包
bash build.sh --pkg
# 编译 host + device 包
bash build.sh --pkg --full
```

若您的编译环境无法访问网络，您需要在联网环境中下载上述开源软件压缩包，并手动上传至编译环境中，并通过 `--cann_3rd_lib_path` 参数指定软件包路径：

```shell
# 指定软件包路径，默认为：./third_party
bash build.sh --cann_3rd_lib_path={your_3rd_party_path}
```

编译完成后会在`./build_out`目录下生成 `cann-hccl_<version>_linux-<arch>.run` 软件包。

> `<version>`表示软件版本号，`<arch>`表示操作系统架构，取值包括“x86_64”与“aarch64”。

## 安装

安装编译生成的HCCL软件包：

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --full
```

请注意：编译时需要将上述命令中的软件包名称替换为实际编译生成的软件包名称。

安装完成后，用户编译生成的HCCL软件包会替换已安装CANN开发套件包中的HCCL相关软件。

## 卸载

卸载已安装的HCCL软件包：

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --uninstall
```

请注意：卸载时需要将上述命令中的软件包名称替换为实际安装的软件包名称。

## LLT 测试

安装完编译生成的HCCL软件包后，可通过如下命令执行LLT用例。

```shell
bash build.sh --ut
```

## 上板测试

HCCL软件包安装完成后，开发者可通过HCCL Test工具进行集合通信功能与性能的测试，HCCL Test工具的使用流程如下：

1. 环境准备

   运行本项目除需安装CANN Toolkit开发套件包外，还需安装NPU驱动、NPU固件和CANN ops算子包。

   NPU驱动和固件可参考[昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)中的“安装NPU驱动和固件”章节进行安装：

   ```shell
   # 安装驱动
   ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --install-for-all

   # 安装固件
   ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
   ```

   CANN ops算子包可根据NPU产品型号和操作系统架构，从[CANN软件包归档页面](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master/)中下载对应的CANN ops包，参考[昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)中的“安装CANN-安装ops算子包”章节进行安装：

   ```shell
   # 安装算子包
   bash Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run --install
   ```

   - `<chip_type>`: 表示NPU产品型号。
   - `<version>`: 表示CANN包版本号。
   - `<arch>`: 表示CPU架构，如aarch64、x86_64。

2. 工具编译

   使用 HCCL Test 工具前需要安装 MPI 依赖，配置相关环境变量，并编译 HCCL Test 工具，详细操作方法可参见配套版本的[昇腾文档中心-HCCL 性能测试工具使用指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest)中的“工具编译”章节。

3. 关闭验签

   - hccl仓编译产生`cann-hccl_<version>_linux-<arch>.run`软件包中含有`aicpu_hccl.tar.gz`（Hccl AICPU 算子包)
   - `aicpu_hccl.tar.gz`会在业务启动时加载至Device，加载过程中默认会由驱动进行安全验签，确保包可信
   - 开发者下载hccl仓源码自行编译产生`aicpu_hccl.tar.gz`并不含签名头，为此需要关闭驱动安全验签的机制
   - 关闭验签方式：

      配套使用HDK 25.5.T2.B001或以上版本，并通过该HDK配套的npu-smi工具关闭验签。参考如下命令，以root用户在物理机上执行。
      以device 0为例：
      ```shell
      npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # 使能验签配置
      npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # 关闭客户自定义验签
      ```

4. 执行HCCL Test测试命令，测试集合通信的功能及性能

   以1个计算节点，8个NPU设备，测试AllReduce算子的性能为例，命令示例如下：

   ```shell
   # “/usr/local/Ascend”是root用户以默认路径安装的CANN软件安装路径，请根据实际情况替换
   cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test

   # 数据量（-b）从8KB到64MB，增量系数（-f）为2倍，参与训练的NPU个数为8
   mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
   ```

   工具的详细使用说明可参见[昇腾文档中心-HCCL 性能测试工具使用指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest)中的“工具执行”章节。

5. 查看结果

   执行完HCCL Test工具后，回显示例如下：

   ![hccltest_result](figures/hccl_test_result.png)

   - “check_result”为 success，代表通信算子执行结果成功，AllReduce 算子功能正确。
   - ”aveg_time“：集合通信算子的执行耗时，单位 us。
   - ”alg_bandwidth“：集合通信算子执行带宽，单位为 GB/s。
   - ”data_size“：单个 NPU 上参与集合通信的数据量，单位为 Bytes。
