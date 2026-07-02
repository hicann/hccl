# Source Code Build

## Environment Preparation

This project supports building from source. Before compiling and running, set up the basic environment and download the source code by following the steps below. Ensure that the NPU driver, firmware, and CANN software are installed.

### Prerequisites

The following software dependencies are required for compiling this project. Ensure that the version requirements are met.

- python >= 3.7.0
- pip3 >= 20.3.0
- gcc and g++: 7.3.0 to 13.3.x
- cmake >= 3.16.0
- ccache (optional, used to improve secondary compilation speed)
- googletest (required only when running UT, recommended version release-1.14.0)

### Installing the CANN Software Package

1. **Install the driver and firmware (runtime dependency)**

   For downloading and installing the driver and firmware, refer to the "Prepare Software Package" and "Install NPU Driver and Firmware" chapters in the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard). The driver and firmware are runtime dependencies. If you only need to compile the source code of this project, you do not need to install them.

2. **Install the CANN software package**

   - **Scenario 1: Experience master version features or develop based on the master version**

     Click the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select the latest version, and download the corresponding software package based on the product model and environment architecture. For installation commands, see the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

     1. Install the CANN Toolkit development kit package.

        ```bash
        # Ensure the installation package has executable permissions
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

     2. Install the CANN ops operator package (runtime dependency).

        The ops operator package is a runtime dependency. If you only need to compile the source code of this project, you do not need to install this package.

        ```bash
        # Ensure the installation package has executable permissions
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

     - \$\{cann_version\}: CANN software package version number.
     - \$\{arch\}: CPU architecture, for example, aarch64 or x86_64.
     - \$\{soc_name\}: NPU model name.
     - \$\{install_path\}: Specified installation path. The CANN ops operator package must be installed in the same path as the CANN Toolkit development kit package. The root user default installation path is `/usr/local/Ascend`.

   - **Scenario 2: Experience released version features or develop based on a released version**

     Visit the [CANN official download center](https://www.hiascend.com/cann/download), select a released version (only CANN 8.5.0 and later versions are supported), download the corresponding software package based on the product model and environment architecture, and follow the commands provided on the webpage to complete the installation.

### Environment Verification

After installing the CANN software package, verify that the environment is functioning correctly.

- **Check NPU devices**:

  ```bash
  # Run npu-smi. If device information is displayed correctly, the driver is working.
  npu-smi info
  ```

- **Check CANN software**:

  ```bash
  # View the version information provided by the version field of the CANN Toolkit development kit package (default path installation). <arch> indicates the CPU architecture (aarch64 or x86_64).
  cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
  # View the version information provided by the version field of the CANN ops operator package (default path installation).
  cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
  ```

### Environment Variable Configuration

Run the appropriate command to apply the environment variables.

```bash
# Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified path installation
# source ${install_path}/cann/set_env.sh
```

## Compilation and Installation

### Downloading the Source Code

Run the following command to download the source code. Replace \$\{tag_version\} with the target branch label name. For the mapping between source code branch labels and CANN versions, see the [release repository](https://gitcode.com/cann/release-management).

```shell
# Download the source code for the corresponding branch
git clone -b ${tag_version} https://gitcode.com/cann/hccl.git
```

### Compiling the Source Code

This project provides one-click compilation and build capabilities. Go to the root directory of the repository and run the following command:

```shell
# Compile the host package
bash build.sh --pkg
# Compile the host and device package
bash build.sh --pkg --full
```

During compilation, the dependency packages listed in [Open-Source Third-Party Software Dependencies](#open-source-third-party-software-dependencies) are automatically downloaded. If the compilation environment cannot access the network, download the dependency packages in an online environment, upload them manually to the compilation environment, and use the `--cann_3rd_lib_path` parameter to specify the path to the dependency packages.

```shell
# Specify the software package path. The default path is: ./third_party
bash build.sh --cann_3rd_lib_path={your_3rd_party_path}
```

After compilation, the `cann-hccl_<version>_linux-<arch>.run` software package is generated in the `./build_out` directory.

`<version>` indicates the software version number, and `<arch>` indicates the operating system architecture. The values include `x86_64` and `aarch64`.

### Installation

Install the compiled HCCL software package:

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --full
```

Replace the software package name in the command with the actual package name.

After installation, the compiled HCCL software package replaces the HCCL-related software in the installed CANN Toolkit development kit package.

### Uninstallation

To uninstall the compiled HCCL software package and restore to the state after installing the CANN Toolkit development kit package, run the following command:

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --uninstall
```

Replace the software package name in the command with the actual package name.

## Testing

### LLT Testing

After installing the compiled HCCL software package, run the following command to execute LLT test cases:

```shell
bash build.sh --ut
```

### On-Device Testing

> [!NOTE] Note
> Before on-device testing, ensure that the driver firmware, CANN Toolkit development kit package, and CANN ops operator package are installed.

Developers can use the HCCL Test tool to test collective communication functionality and performance. The procedure for using the HCCL Test tool is as follows:

1. Tool compilation

   Before using the HCCL Test tool, install the MPI dependency and compile the HCCL Test tool. For detailed operations, see the "MPI Installation and Configuration" and "Tool Compilation" chapters in the corresponding version of the [Ascend Documentation Center - HCCL Performance Test Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolHcclTest).

2. Disable signature verification

   The `cann-hccl_<version>_linux-<arch>.run` software package generated by this source repository contains `aicpu_hccl.tar.gz` (the HCCL AICPU operator package).

   `aicpu_hccl.tar.gz` is loaded to the Device when the service starts. During loading, the driver performs security signature verification by default to ensure that the package is trustworthy. Because the `aicpu_hccl.tar.gz` package compiled by developers from this source repository does not contain a signature header, disable the driver security signature verification mechanism.

   **Method to disable signature verification:**

   Use Ascend HDK 25.5.T2.B001 or later, and use the npu-smi tool provided with the Ascend HDK to disable signature verification. The following reference commands must be executed as the root user on the physical machine (using device 0 as an example).

   ```shell
   npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # Enable signature verification configuration
   npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # Disable customer custom signature verification
   ```

3. Run the HCCL Test command to test the functionality and performance of collective communication.

   The following example tests the performance of the AllReduce operator on one compute node with 8 NPU devices:

   ```shell
   # /usr/local/Ascend is the CANN software installation path for the root user using the default path. Replace it with the actual path.
   cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test

   # Data size (-b) from 8 KB to 64 MB, increment factor (-f) of 2, number of NPUs participating in training: 8
   mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
   ```

   For detailed usage instructions, see the "Tool Execution" chapter in the [Ascend Documentation Center - HCCL Performance Test Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolHcclTest).

4. View the results.

   After running the HCCL Test tool, the following example output is displayed:

   ![hccltest_result](./figures/hccl_test_result.png)

   - `check_result` = success indicates that the communication operator executed successfully, and the AllReduce operator functions correctly.
   - `aveg_time`: Execution time of the collective communication operator, in microseconds.
   - `alg_bandwidth`: Execution bandwidth of the collective communication operator, in GB/s.
   - `data_size`: Data volume on a single NPU participating in collective communication, in Bytes.

## Appendix

### Open-Source Third-Party Software Dependencies

When compiling this project, the following third-party open-source software is required:

| Open-Source Software | Version | Download URL |
| ------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| googletest | 1.14.0 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake | master-001 | [cmake-master-001.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-001.tar.gz) |
