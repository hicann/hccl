# 源码构建

## 环境准备

本项目支持源码构建，编译运行前需参考以下步骤完成基础环境搭建和源码下载，确保已安装NPU驱动、固件和CANN软件。

### 前置依赖

本项目编译用到的软件依赖如下，注意满足版本号要求。

- python >= 3.7.0
- pip3 >= 20.3.0
- gcc & g++ : 7.3.0 至 13.3.x
- cmake >= 3.16.0
- ccache（可选，用于提高二次编译速度）
- googletest（仅执行UT时依赖，建议版本 release-1.14.0）

### 安装CANN软件包

1. **安装驱动与固件（运行态依赖）**

    驱动与固件的下载和安装操作请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中“准备软件包”和“安装NPU驱动和固件”章节。驱动与固件是运行态依赖，若仅编译本项目源码，可以不安装。

2. **安装CANN软件包**

    - **场景1：体验master版本能力或基于master版本进行开发**

        请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，选择最新时间版本，并根据产品型号和环境架构下载对应软件包。安装命令如下，更多指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

        1. 安装CANN Toolkit开发套件包。

            ```bash
            # 确保安装包具有可执行权限
            chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
            # 安装命令
           ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
           ```

        2. 安装CANN ops算子包（运行态依赖）。

            ops算子包是运行态依赖，若仅编译本项目源码，可不安装此软件包。
    
            ```bash
            # 确保安装包具有可执行权限
            chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
            # 安装命令
            ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```
    
        - \$\{cann\_version\}：表示CANN软件包版本号。
        - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
        - \$\{soc\_name\}：表示NPU型号名称。
        - \$\{install\_path\}：表示指定安装路径，CANN ops算子包需与CANN Toolkit开发套件包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。

    - **场景2：体验已发布版本能力或基于已发布版本进行开发**

        请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择发布版本（仅支持CANN 8.5.0及后续版本），并根据产品型号和环境架构下载对应软件包，最后参考网页提供的命令完成安装。

### 环境验证

安装完CANN软件包后，需验证环境是否正常。

- **检查NPU设备**：

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```

- **检查CANN软件**：

   ```bash
    # 查看CANN Toolkit开发套件包的version字段提供的版本信息（默认路径安装），<arch>表示CPU架构（aarch64或x86_64）。
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
    # 查看CANN ops算子包的version字段提供的版本信息（默认路径安装）。
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
   ```

### 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

## 编译安装

### 下载源码

源码下载命令如下，请将\$\{tag\_version\}替换为目标分支标签名，源码分支标签与CANN版本配套关系参见[release仓库](https://gitcode.com/cann/release-management)。

```shell
# 下载项目对应分支源码
git clone -b ${tag_version} https://gitcode.com/cann/hccl.git
```

### 编译源码

本项目提供一键式编译构建能力，进入代码仓根目录，执行如下命令：

```shell
# 编译 host 包
bash build.sh --pkg
# 编译 host + device 包
bash build.sh --pkg --full
```

编译时会自动下载[开源第三方软件依赖](#开源第三方软件依赖)中列出的依赖包。如果编译环境无法访问网络，您需要在联网环境中下载上述依赖压缩包，手动上传至编译环境，并通过 `--cann_3rd_lib_path` 参数指定依赖包的存放路径。

```shell
# 指定软件包路径，默认为：./third_party
bash build.sh --cann_3rd_lib_path={your_3rd_party_path}
```

编译完成后会在`./build_out`目录下生成 `cann-hccl_<version>_linux-<arch>.run` 软件包。

其中`<version>`表示软件版本号，`<arch>`表示操作系统架构，取值包括“x86_64”与“aarch64”。

### 安装

安装编译生成的HCCL软件包：

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --full
```

请注意：编译时需要将上述命令中的软件包名称替换为实际软件包名称。

安装完成后，用户编译生成的HCCL软件包会替换已安装CANN Toolkit开发套件包中的HCCL相关软件。

### 卸载

若您想卸载编译生成的HCCL软件包，恢复到安装完CANN Toolkit开发套件包的状态，可参考如下命令：

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --uninstall
```

请注意：卸载时需要将上述命令中的软件包名称替换为实际软件包名称。

## 测试

### LLT 测试

安装完编译生成的HCCL软件包后，可通过如下命令执行LLT用例。

```shell
bash build.sh --ut
```

### 上板测试

> [!NOTE]说明
> 上板测试前，请确保已安装驱动固件、CANN Toolkit开发套件包与CANN ops算子包。

开发者可通过HCCL Test工具进行集合通信功能与性能的测试，HCCL Test工具的使用流程如下：

1. 工具编译

   使用HCCL Test工具前需要安装MPI依赖、编译HCCL Test工具，详细操作方法可参见配套版本的[昇腾文档中心-HCCL性能测试工具使用指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest)中的“MPI安装与配置”与“工具编译”章节。

2. 关闭验签

   源码仓编译生成的`cann-hccl_<version>_linux-<arch>.run`软件包中包含`aicpu_hccl.tar.gz`（HCCL AICPU 算子包)。

   `aicpu_hccl.tar.gz`会在业务启动时加载至Device，加载过程中默认会由驱动进行安全验签，确保包可信。由于开发者通过本源码仓自行编译生成的`aicpu_hccl.tar.gz`包中并不含签名头，所以需要关闭驱动安全验签的机制。

   **关闭验签方式：**

      配套使用Ascend HDK 25.5.T2.B001及以上版本，并通过该Ascend HDK自带的npu-smi工具关闭验签。以下为参考命令，需要以root用户在物理机上执行（以device 0为例）。

      ```shell
      npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # 使能验签配置
      npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # 关闭客户自定义验签
      ```

3. 执行HCCL Test测试命令，测试集合通信的功能及性能。

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

   ![hccltest_result](./figures/hccl_test_result.png)

   - “check_result”为 success，代表通信算子执行结果成功，AllReduce 算子功能正确。
   - ”aveg_time“：集合通信算子的执行耗时，单位 us。
   - ”alg_bandwidth“：集合通信算子执行带宽，单位为 GB/s。
   - ”data_size“：单个 NPU 上参与集合通信的数据量，单位为 Bytes。

## 附录

### 开源第三方软件依赖

编译本项目时，依赖的第三方开源软件列表如下：

| 开源软件      | 版本                   | 下载地址                                                                                                                                                                                                    |
| ------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| makeself      | 2.5.0                  | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)                                     |
| googletest    | 1.14.0                 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz)|
| cann-cmake     | master-001            | [cmake-master-001.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-001.tar.gz) |
