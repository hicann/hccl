# 源码构建

## 环境准备

1. 安装依赖。

   HCCL集合通信库编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - ccache
   - nlohmann_json
   - googletest（仅执行UT时依赖，建议版本 release-1.14.0）

   nlohmann_json和googletest可通过`build_third_party.sh`进行编译安装，可通过`--output_path`参数指定安装目录，默认为`./output/third_party`：

   ```shell
   bash build_third_party.sh --output_path=${THIRD_LIB_PATH}
   ```

2. 安装社区尝鲜版CANN Toolkit包

    编译本项目依赖CANN开发套件包（cann-toolkit），请根据操作系统架构，下载对应的CANN Toolkit安装包，参考[昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)进行安装：

    - aarch64架构：[Ascend-cann-toolkit_8.5.0_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/hccl/Ascend-cann-toolkit_8.5.0_linux-aarch64.run)

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

本项目提供一键式编译构建能力，进入代码仓根目录，执行如下命令：

```shell
bash build.sh --pkg
```

编译完成后会在`./build_out`目录下生成 `cann-hccl_<version>_linux-<arch>.run` 软件包。

> `<version>`表示软件版本号，`<arch>`表示操作系统架构，取值包括“x86_64”与“aarch64”。

## 安装

安装编译生成的HCCL软件包：

```shell
bash ./build_out/cann-hccl<version>_linux-<arch>.run --full
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

   运行本项目除需安装CANN Toolkit开发套件包外，还需安装Ascend HDK包、Ascend-ops算子包，下载链接如下，安装方式可参考[昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)进行安装：

   - Atlas A2系列产品:
     - Ascend HDK驱动包：[Ascend-hdk-910b-npu-driver_25.5.0.b061_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/hccl/Ascend-hdk-910b-npu-driver_25.5.0.b061_linux-aarch64.run)
     - Ascend HDK固件包：[Ascend-hdk-910b-npu-firmware_7.8.0.5.201.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/hccl/Ascend-hdk-910b-npu-firmware_7.8.0.5.201.run)
     - Ascend-ops包（aarch64架构）：[Ascend-cann-ops-910b_8.5.0_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/hccl/Ascend-cann-ops-910b_8.5.0_linux-aarch64.run)

   - Atlas A3系列产品:
     - Ascend HDK驱动包：[Atlas-A3-hdk-npu-driver_25.5.0.b061_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/hccl/Atlas-A3-hdk-npu-driver_25.5.0.b061_linux-aarch64.run)
     - Ascend HDK固件包：[Atlas-A3-hdk-npu-firmware_7.8.0.5.201.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/hccl/Atlas-A3-hdk-npu-firmware_7.8.0.5.201.run)
     - Ascend-ops包（aarch64架构）：[Atlas-A3-cann-ops_8.5.0_linux-aarch64.run](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/hccl/Atlas-A3-cann-ops_8.5.0_linux-aarch64.run)

2. 工具编译

   使用 HCCL Test 工具前需要安装 MPI 依赖，配置相关环境变量，并编译 HCCL Test 工具，详细操作方法可参见配套版本的[昇腾文档中心-HCCL 性能测试工具使用指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest)中的“工具编译”章节。

3. 执行HCCL Test测试命令，测试集合通信的功能及性能

   以1个计算节点，8个NPU设备，测试AllReduce算子的性能为例，命令示例如下：

   ```shell
   # “/usr/local/Ascend”是root用户以默认路径安装的CANN软件安装路径，请根据实际情况替换
   cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test

   # 数据量（-b）从8KB到64MB，增量系数（-f）为2倍，参与训练的NPU个数为8
   mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
   ```

   工具的详细使用说明可参见[昇腾文档中心-HCCL 性能测试工具使用指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest)中的“工具执行”章节。

4. 查看结果

   执行完HCCL Test工具后，回显示例如下：

   ![hccltest_result](figures/hccl_test_result.png)

   - “check_result”为 success，代表通信算子执行结果成功，AllReduce 算子功能正确。
   - ”aveg_time“：集合通信算子的执行耗时，单位 us。
   - ”alg_bandwidth“：集合通信算子执行带宽，单位为 GB/s。
   - ”data_size“：单个 NPU 上参与集合通信的数据量，单位为 Bytes。
