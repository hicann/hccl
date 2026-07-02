# Custom Communication Operator - AllGather

## Sample Description

This sample demonstrates how to develop an AllGather communication operator based on the HCCL communication programming interface. It covers the following features:

1. Implement an AllGather collective communication operator based on the CCU_SCHED communication engine.
2. Support independent building and deployment of custom operator packages.

## Directory Structure

```text
├── CMakeLists.txt                      # Compilation and build configuration file
├── op_host/
|   ├── allgather.cc                    # HcclAllGatherCustom operator implementation source file
|   ├── utils.cc                        # Utility module (channel acquisition, thread acquisition, Kernel registration)
|   └── utils.h                         # Utility module header file
├── op_kernel_ccu/
|   ├── ccu_kernel.cc                   # CCU Kernel implementation logic
|   ├── ccu_kernel.h                    # CCU Kernel header file
|   ├── exec_op.cc                      # CCU operator orchestration logic
|   └── exec_op.h                       # CCU operator orchestration header file
├── inc/
|   ├── hccl_custom_allgather.h         # Custom AllGather operator interface header file
|   ├── common.h                        # Common type header file
|   └── log.h                           # Log macro definitions
└── testcase/
    ├── main.cc                         # Sample implementation source file
    └── Makefile                        # Compilation and build configuration file
```

> The custom operator compilation project depends on the [cmake](../../../cmake) configuration and the [build.sh](../../../build.sh) compilation script in the HCCL repository.
> 
> - cmake contains CMake configuration, MakeSelf packaging configuration, and so on.
> - build.sh is the compilation entry for the project.

## 1. Environment Preparation

### 1.1 Environment Requirements

This sample supports the following Ascend products in a single-server N-card configuration (N >= 2):

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

The following software dependencies are required for compiling this sample. Ensure that the version requirements are met:

- gcc and g++ >= 7.3.0
- cmake >= 3.16.0

### 1.2 Install the CANN Toolkit Development Kit Package

Refer to the [Ascend Documentation Center - CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard) to install the CANN Toolkit development kit package version 9.1.0 that matches the sample.

The installation supports both default path and specified path installation:

- **Default path installation**: Do not specify the `--install-path` parameter. After installation, the software is located at `/usr/local/Ascend` (root user) or `${HOME}/Ascend` (non-root user).
- **Specified path installation**: Use the `--install-path` parameter to specify the installation directory. After installation, the software is located at `${install_path}`. Use this path for subsequent compilation and environment variable configuration.

```bash
# Ensure the installation package has executable permissions
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
# Default path installation
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install
# Specified path installation (replace ${install_path} with the actual installation directory)
# ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
```

### 1.3 Configure Environment Variables

After installing the CANN Toolkit development kit package, select the corresponding environment variable configuration command based on the installation method to apply the environment variables.

```bash
# Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified path installation. ${install_path} indicates the actual CANN-Toolkit package installation path.
# source ${install_path}/cann/set_env.sh
```

`set_env.sh` is an environment initialization script that configures environment variables such as PATH, LD_LIBRARY_PATH, and PYTHONPATH, enabling the compiler and runtime to correctly find the Ascend NPU toolchain, library files, and operator libraries.

## 2. Compiling the Custom Operator Package

The HCCL repository provides a custom operator compilation and packaging project. This project depends on the following files in the repository:

```text
├── build.sh                        # Compilation entry in the hccl repository root directory
├── CMakeLists.txt                  # Compilation and build configuration file in the hccl repository root directory
├── cmake/
|   ├── config.cmake                # CMake variable definitions
|   ├── func.cmake                  # CMake function definitions
|   ├── package.cmake               # Signature and packaging function definitions
|   └── makeself_custom.cmake       # MakeSelf packaging logic
└── scripts/
    ├── custom/install.sh           # Custom operator package installation script
    └── sign/add_header_sign.py     # Operator package signing script
```

Therefore, developers first need to download the hccl repository, then run `build.sh` from the repository root directory for compilation, specifying the custom operator project path using `custom_ops_path`:

```bash
# Download the hccl repository
git clone https://gitcode.com/cann/hccl.git

# Compile the custom operator package
cd hccl
bash build.sh --vendor=cust --ops=allgather_ccu --custom_ops_path=./examples/05_custom_ops_allgather/ccu
```

> Where:
> 
> - `--vendor` specifies the custom operator identifier.
> - `--ops` specifies the custom operator name.
> - `--custom_ops_path` specifies the custom operator project path.

## 3. Installing the Custom Operator Package

The custom operator installation package is located in the `./build_out` directory. Install it using the `--install` parameter:

```bash
./build_out/cann-hccl_custom_allgather_ccu_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> Where:
> 
> - `<arch>` is the system architecture of the current compilation environment.
> - `<ascend_cann_path>` is an optional parameter indicating the CANN software package installation directory. The default value is the CANN software package path where the `ASCEND_CUSTOM_OPP_PATH` or `ASCEND_OPP_PATH` environment variable is located.

The custom operator package installation information is as follows:

- Header file: `${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_allgather.h`
- Dynamic library: `${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_allgather.so`

> `${ASCEND_HOME_PATH}` is the CANN-Toolkit installation path.

## 4. Running the Custom Operator

### 4.1 Compile the Test Sample

Run the following commands in the `examples/05_custom_ops_allgather/ccu/testcase` directory:

```bash
# Compile the test sample
make
```

### 4.2 Run the Test Sample

Run the following commands in the `examples/05_custom_ops_allgather/ccu/testcase` directory:

```bash
# Run the test sample
make test

# Or run the sample binary directly
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH}
./custom_allgather_ccu
```

### 4.3 Sample Output

The input data of all nodes is initialized to the Device ID of that node. After successful execution, the terminal displays log output similar to the following (using 2 cards as an example):

```text
Found 2 NPU device(s) available
rankId: 1, input: [ 1 ]
rankId: 0, input: [ 0 ]
rankId: 0, output: [ 0 1 ]
rankId: 1, output: [ 0 1 ]
```
