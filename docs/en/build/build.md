# Source Code Build

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-11T01:57:20.254Z pushedAt=2026-08-11T03:17:03.201Z -->

## Environment Preparation

This project supports source code build. Before build and run, complete basic environment setup and source code download by following the steps below. Ensure that the NPU driver, firmware, and CANN software are installed.

### Prerequisites

The following lists the software dependencies for building this project. Ensure that the version requirements are met.

- Python >= 3.7.0

- pip3 >= 20.3.0

- gcc & g++ : 7.3.0 to 13.3.x

- CMake >= 3.16.0

- ccache (optional, used to speed up recompilation)

- GoogleTest (only required when running UTs; release-1.14.0 recommended)

### Installing the CANN Software Package

1. **Install the Driver and Firmware (Runtime Dependency)**

    For downloading and installing the driver and firmware, see "Preparing Software Packages" and "Installing the NPU Driver and Firmware" in *[CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)*. The driver and firmware are runtime dependencies. If you only need to compile the source code of this project, you can skip this step.

2. **Install the CANN Software Package**

    - **Use case 1: Try the master version or develop based on the master version**

        Click [here](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select the latest version, and download the corresponding software package based on the product model and environment architecture. The installation commands are as follows. For more instructions, see *[CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)*.

        1. Install the CANN Toolkit.

            ```bash
            # Ensure you have executable permissions on the installation package.
            chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
            # Installation command
           ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
           ```

        2. Install the CANN ops package (runtime dependency).

            This package is not required if you only compile the source code of this project.

            ```bash
            # Ensure you have executable permissions on the installation package.
            chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
            # Installation command
            ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        - ${cann_version}: CANN software package version number.

        - ${arch}: CPU architecture, such as aarch64 or x86_64.

        - \$\{soc_name\}: NPU model.

        - \$\{install_path\}: specified installation path. The CANN ops package must be installed in the same path as the CANN Toolkit. For user `root`, the default installation path is `/usr/local/Ascend`.

    - **Use case 2: Try a released version or develop based on a released version**

        Visit the [CANN download center](https://www.hiascend.com/cann/download), select a release (only CANN 8.5.0 and later versions are supported), download the corresponding software package based on the product model and environment architecture, and then complete the installation by following the commands provided on the webpage.

### Environment Verification

After installing the CANN software packages, verify that the environment is functional.

- **Check the NPU**:

    ```bash
    # Run npu-smi. If the device information is displayed properly, the driver is normal.
    npu-smi info
    ```

- **Check the CANN software**:

   ```bash
    # View the version information provided by the version field of the CANN Toolkit (installed to the default path). arch indicates the CPU architecture (aarch64 or x86_64).
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
    # View the version information provided by the version field of the CANN ops package (installed to the default path).
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
   ```

### Environment Variable Configuration

Choose the appropriate command to make the environment variables take effect.

```bash
# Installation to the default path, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Installation to a custom path
# source ${install_path}/cann/set_env.sh
```

## Build and Installation

### Downloading Source Code

The source code download command is as follows. Replace `${tag_version}` with the target branch tag name. For the mapping between source code branch tags and CANN versions, see the [release repository](https://gitcode.com/cann/release-management).

```shell
# // Download the source code of the corresponding project branch.
git clone -b ${tag_version} https://gitcode.com/cann/hccl.git
```

### Building Source Code

This project allows one-click build. Go to the root directory of the repository and run the following commands:

```shell
# Build the host package.
bash build.sh --pkg
# Build the host and device package.
bash build.sh --pkg --full
```

During build, the dependency packages listed in [open-source third-party software dependencies](#open-source-third-party-software-dependencies) are automatically downloaded. If the build environment cannot access the network, you need to download the preceding dependency packages in an environment with network connection, manually upload them to the build environment, and specify the storage path of the dependency packages using `--cann_3rd_lib_path`.

```shell
# Specify the software package path. Default: ./third_party.
bash build.sh --cann_3rd_lib_path={your_3rd_party_path}
```

After build, the `cann-hccl_<version>_linux-<arch>.run` software package is generated in the `./build_out` directory.

Here, `<version>` indicates the software version number, and `<arch>` indicates the OS architecture, with values including `x86_64` and `aarch64`.

### Installation

Install the built HCCL software package:

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --full
```

Note: During build, replace the software package name in the preceding command with the actual name.

After installation, the user-built HCCL software package replaces the HCCL-related software in the installed CANN Toolkit package.

### Uninstall

If you want to uninstall the built HCCL software package and restore the state as after installing the CANN Toolkit package, run the following command:

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --uninstall
```

Note: When uninstalling, replace the software package name in the preceding command with the actual name.

## Test

### LLT Test

After installing the built HCCL software package, execute LLT test cases using the following command.

```shell
bash build.sh --ut
```

### On-Board Test

> [!NOTE] Note
> Before the on-board test, ensure that the driver and firmware, CANN Toolkit package, and CANN ops package have been installed.

You can use HCCL Test to test collective communication functionality and performance as follows:

1. Tool build

   Before using HCCL Test, you need to install MPI dependencies and build HCCL Test. For details, see "Installing and Configuring MPI" and "Compilation" in the corresponding version of [HCCL Performance Test Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolHcclTest).

2. Disable signature verification.

   The `cann-hccl_<version>_linux-<arch>.run` software package built from the source repository contains `aicpu_hccl.tar.gz` (HCCL AICPU operator package).

   `aicpu_hccl.tar.gz` is loaded to the device when the service starts. During loading, the driver performs security signature verification by default to ensure the package is trusted. Since the user-built `aicpu_hccl.tar.gz` package from this source repository does not contain a signature header, the driver security signature verification mechanism must be disabled.

   **To disable signature verification:**

      Use Ascend HDK 25.5.T2.B001 or later, and disable signature verification using the npu-smi tool bundled with the Ascend HDK. The following is a reference command, which must be executed by the root user on the physical machine (using device 0 as an example).

      ```shell
      npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # Enable signature verification.
      npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # Disable custom signature verification.
      ```

3. Run the HCCL Test command to test the functionality and performance of collective communication.

   The following example tests the performance of the AllReduce operator with one compute node and eight NPUs:

   ```shell
   # "/usr/local/Ascend" is the default CANN installation path for the root user. Replace it with the actual path.
   cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test

   # Data size (-b) from 8 KB to 64 MB, increment factor (-f) is 2x, and the number of NPUs participating in training is 8.
   mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
   ```

   For detailed tool instructions, see "Execution" in [HCCL Performance Test Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolHcclTest).

4. View the results.

   After HCCL Test finishes execution, the output is displayed as follows:

   ![hccltest_result](./figures/hccl_test_result.png)

   - `check_result`: success, indicating that the communication operator is executed and the AllReduce operator functions properly.

   - `aveg_time`: execution time of the collective communication operator, in μs.

   - `alg_bandwidth`: execution bandwidth of the collective communication operator, in GB/s.

   - `data_size`: amount of data participating in collective communication on a single NPU, in bytes.

## Reference

### Open-Source Third-Party Software Dependencies

The following table lists the open-source third-party software on which this project depends during build:

| Software | Version | Download |
| -------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| makeself             | 2.5.0   | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)                                     |
| googletest           | 1.14.0  | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz)                                                                          |
| cann-cmake           | master-001 | [cmake-master-001.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-001.tar.gz)
