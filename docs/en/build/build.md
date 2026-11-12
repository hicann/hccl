# Source Code Build

This topic describes how to build, install, and test the HCCL source code. For environment preparation, you can choose [Docker Deployment](#scenario-1-docker-deployment) or [Host Deployment](#scenario-2-host-deployment).

## Environment Preparation

This project supports source code build. Before building, complete the basic environment setup and source code download by following the steps below, and install the CANN Toolkit package. The NPU driver, firmware, and CANN ops package are runtime dependencies. If you only compile the source code of this project, you can skip installing them; however, install them before running or on-board testing.

### Prerequisites

The following lists the software dependencies for building this project. Ensure that the version requirements are met.

- Python >= 3.7.0
- pip3 >= 20.3.0
- gcc & g++ : 7.3.0 to 14.2.x
- Cmake >= 3.16.0
- ccache (optional, used to speed up recompilation)
- GoogleTest (only required when running UTs; release-1.14.0 recommended)

### Scenario 1: Docker Deployment

**1. Install the driver and firmware**: Install Docker on the host machine, and install the matching driver and firmware by referring to the *[CANN Software Installation Guide](https://www.hiascend.com/en/document/redirect/CannCommunityInstWizard)*.

**2. Download the build image**: The A3 image is used as an example below. Select and run one command based on the CPU architecture of the host machine.

```bash
# x86_64
docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-a3-ubuntu22.04-py3.12-devel
# aarch64
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-a3-ubuntu22.04-py3.12-devel
```

The image comes with build tools and CANN software pre-installed, and you enter the container as user `root` by default. For other products and versions, select an image in [Ascend-CANN Image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884). The image version must match the source code branch, driver, and firmware. When developing the master version or changing the CANN version, you can update CANN in the container by referring to [Installing the CANN Software Package](#installing-the-cann-software-package).

**3. Create and enter the container**: The following assumes that `/dev/davinci0` and `/dev/davinci1` are used and the driver is installed in `/usr/local/Ascend`.

```bash
image=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-a3-ubuntu22.04-py3.12-devel
docker run --name env_for_hccl_build --network host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -it "${image}" bash
```

Adjust the device numbers and mount paths according to the actual environment. When compiling only, the device and driver-related mounts can be omitted; retain them for on-board testing. For more options, run `docker run --help`.

#### A5 (Atlas 950) Container Configuration

A5 uses a matching image and mounts `/dev/ummu`, `/dev/uburma`, and the driver `topo` directory to support topology discovery and UB communication; `/dev/devmm_svm` does not need to be mounted. Two devices are still used as an example below:

```bash
image=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-openeuler24.03-py3.12
docker run --name env_for_hccl_build_a5 --network host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/ummu \
  --device /dev/uburma \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools \
  -v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -it "${image}" bash
```

#### Building and Testing in the Container

After entering the container, load the CANN environment variables. The script path depends on the actual installation location of the selected image:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Then proceed with [Build and Installation](#build-and-installation) and [Test](#test). Before using HCCL Test, prepare MPI and build the tool in the container by following the matching [Tool Guide](https://www.hiascend.com/en/document/redirect/CannCommunityToolHcclTest); the test devices must be consistent with the container mapping. If you directly test the HCCL bundled with the image, you do not need to rebuild HCCL or disable signature verification.

### Scenario 2: Host Deployment

#### Installing the CANN Software Package

1. **Install the driver and firmware (runtime dependency)**

    For downloading and installing the driver and firmware, see "Preparing Software Packages" and "Installing the NPU Driver and Firmware" in *[CANN Software Installation Guide](https://www.hiascend.com/en/document/redirect/CannCommunityInstWizard)*. The driver and firmware are runtime dependencies. If you only need to compile the source code of this project, you can skip this step.

2. **Install the CANN software package**

    - **Use case 1: Try the master version capabilities or develop based on the master version**

        Click the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select the latest version, and download the corresponding software package based on the product model and environment architecture. The installation commands are as follows. For more instructions, see *[CANN Software Installation Guide](https://www.hiascend.com/en/document/redirect/CannCommunityInstWizard)*.

        1. Install the CANN Toolkit.

            ```bash
            # Ensure that the installation package has the execute permission.
            chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
            # Installation command
            ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        2. Install the CANN ops package (runtime dependency).

            The CANN ops package is a runtime dependency. If you only compile the source code of this project, you can skip installing this package.

            ```bash
            # Ensure that the installation package has the execute permission.
            chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
            # Installation command
            ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        - \$\{cann_version\}: CANN software package version number.
        - \$\{arch\}: CPU architecture, such as aarch64 or x86_64.
        - \$\{soc_name\}: NPU model name.
        - \$\{install_path\}: specified installation path. The CANN ops package must be installed in the same path as the CANN Toolkit. For user `root`, the default installation path is `/usr/local/Ascend`.

    - **Use case 2: Try the capabilities of a released version or develop based on a released version**

        Visit the [CANN download center](https://www.hiascend.com/en/cann/download), select a release (only CANN 8.5.0 and later versions are supported), download the corresponding software package based on the product model and environment architecture, and then complete the installation by following the commands provided on the webpage.

#### Environment Verification

After installing the CANN software packages, verify that the environment is functional.

- **Check NPU devices**:

    ```bash
    # Run npu-smi. If device information is displayed, the driver is normal.
    npu-smi info
    ```

- **Check the CANN software**:

   ```bash
    # Check the version information specified by the version field of the CANN Toolkit package (installed to the default path). <arch> indicates the CPU architecture (aarch64 or x86_64).
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
    # View the version information specified by the version field of the CANN ops package (installed to the default path).
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
   ```

#### Environment Variable Configuration

Choose the appropriate command to make the environment variables take effect.

```bash
# Install to the default path, using the root user as an example (for a non-root user, replace /usr/local with ${HOME}).
source /usr/local/Ascend/cann/set_env.sh
# Install to a specified path.
# source ${install_path}/cann/set_env.sh
```

## Build and Installation

### Downloading Source Code

The source code download command is as follows. Replace `${tag_version}` with the target branch tag name. For the mapping between source code branch tags and CANN versions, see the [release repository](https://gitcode.com/cann/release-management).

```shell
# Download the source code of the corresponding project branch.
git clone -b ${tag_version} https://gitcode.com/cann/hccl.git
```

### Building Source Code

This project allows one-click build. Go to the root directory of the repository and run the following commands:

```shell
# Build the host package.
bash build.sh --pkg
# Build the host + device packages.
bash build.sh --pkg --full
```

During build, the dependency packages listed in [Open-Source Third-Party Software Dependencies](#open-source-third-party-software-dependencies) are automatically downloaded. If the build environment cannot access the network, you need to download the preceding dependency packages in an environment with network connection, manually upload them to the build environment, and specify the storage path of the dependency packages using `--cann_3rd_lib_path`.

```shell
# Specify the software package path. Default: ./third_party
bash build.sh --cann_3rd_lib_path={your_3rd_party_path}
```

After build, the `cann-hccl_<version>_linux-<arch>.run` software package is generated in the `./build_out` directory.

Here, `<version>` indicates the software version number, and `<arch>` indicates the OS architecture, with values including `x86_64` and `aarch64`.

### Build Options

The following table lists the common build options supported by `build.sh`. You can combine them as required.

| Option | Description |
| --- | --- |
| `--pkg` | Build the host package (default build target). |
| `--full` | Build the host + device packages, that is, build the host package together with device-side components such as the AICPU kernel. |
| `--aicpu` | Same as `--full`, builds the host + device packages. |
| `--static` | Enable static library build mode. Produces `libhccl_static.a` and packages it as `cann-hccl-static_<version>_linux-<arch>.tar.gz`. |
| `--experimental` | Build the experimental features in the `experimental/` directory (disabled by default). |
| `--custom_ops_path=<PATH>` | Specify the path of the custom operator project to build a custom operator package. Used together with `--ops=<OPS>` and `--vendor=<VENDOR>`. |
| `--build_aarch` | Cross-compile for the aarch64 architecture using the aarch64 compiler in the CANN toolchain. |
| `--asan` | Enable AddressSanitizer to locate issues such as memory out-of-bounds access and leaks. |
| `-u`/`--ut` | Build and run unit tests (UTs). |
| `-s`/`--st` | Build and run system tests (STs). |
| `--cann_3rd_lib_path=<PATH>` | Specify the storage path of third-party dependency packages. Default: `./third_party`. |
| `-j<N>` | Set the number of parallel build threads. The default is twice the number of CPU cores. |

Examples:

```shell
# Static library build.
bash build.sh --pkg --static
# Build and run UTs.
bash build.sh --pkg --ut
# Cross-compile for aarch64 and build the host + device packages.
bash build.sh --pkg --full --build_aarch
# Enable ASAN and run STs.
bash build.sh --pkg --st --asan
# Specify a custom operator project.
bash build.sh --pkg --custom_ops_path={your_custom_ops_path}
```

> [!NOTE]
> To view options not listed in the preceding table, run `bash build.sh --help`.

### Installation

Install the built HCCL software package:

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --full
```

Note: When building, replace the software package name in the preceding command with the actual software package name.

After installation, the user-built HCCL software package replaces the HCCL-related software in the installed CANN Toolkit package.

### Uninstall

If you want to uninstall the built HCCL software package and restore the state as after installing the CANN Toolkit package, run the following command:

```shell
bash ./build_out/cann-hccl_<version>_linux-<arch>.run --uninstall
```

Note: When uninstalling, replace the software package name in the preceding command with the actual software package name.

## Test

### LLT Test

After installing the built HCCL software package, execute LLT test cases using the following command.

```shell
bash build.sh --ut
```

### On-Board Test

> [!NOTE]
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
      npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # Enable the signature verification configuration
      npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # Configure no-signature-verification mode
      ```

3. Run the HCCL Test command to test the functionality and performance of collective communication.

   The following example tests the performance of the AllReduce operator with one compute node and eight NPUs:

   ```shell
   # "/usr/local/Ascend" is the default CANN software installation path for the root user. Replace it with the actual path.
   cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test

   # Data size (-b) ranges from 8 KB to 64 MB, with an increment factor (-f) of 2 and 8 NPUs participating in training.
   mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
   ```

   For detailed tool instructions, see section "Execution" in [HCCL Performance Test Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolHcclTest).

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

| Open Source Software | Version | Download URL |
| ------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| makeself      | 2.5.0                  | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)                                     |
| googletest    | 1.14.0                 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz)|
| cann-cmake     | master-058            | [cmake-master-058.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-058.tar.gz) |
