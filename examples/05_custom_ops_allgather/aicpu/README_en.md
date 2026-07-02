# Custom Communication Operator - AllGather

## Sample Description

This sample demonstrates how to develop an AllGather communication operator based on the HCCL communication programming interface. It covers the following features:

1. Implement an AllGather collective communication operator based on the AICPU communication engine.
2. Support independent building and deployment of custom operator packages.

## Directory Structure

```text
├── CMakeLists.txt                      # Compilation and build configuration file
├── op_host/
|   ├── allgather.cc                    # HcclAllGatherCustom operator implementation source file
|   ├── load_kernel.cc                  # AICPU Kernel loading logic on the Host side
|   ├── launch_kernel.cc                # AICPU Kernel submission logic on the Host side
|   ├── launch_kernel_asc.asc           # ASC syntax Kernel submission wrapper
|   └── utils.cc                        # Utility module
├── op_kernel_aicpu/
|   ├── libcustom_allgather_aicpu_kernel.json  # AICPU Kernel operator description file
|   ├── aicpu_kernel.cc                 # AICPU Kernel implementation logic (C++ mode)
|   ├── aicpu_kernel_asc.aicpu          # AICPU Kernel implementation logic (ASC mode)
|   └── exec_op.cc                      # AICPU operator orchestration logic
├── inc/
|   ├── hccl_custom_allgather.h         # Custom AllGather operator interface header file
|   ├── common.h                        # Common type header file
|   └── log.h                           # Log macro definitions
├── scripts/
|   └── hccl_custom_allgather_aicpu_check_cfg.xml  # Signature configuration file
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

- gcc and g++: 7.3.0 to 13.3.x
- cmake >= 3.16.0

### 1.2 Install the CANN Toolkit Development Kit Package

Refer to the [Ascend Documentation Center - CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard) to install the latest version of the CANN Toolkit development kit package.

### 1.3 Configure Environment Variables

Run the appropriate command to apply the environment variables.

```bash
# Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified path installation. ${install_path} indicates the actual CANN-Toolkit package installation path.
source ${install_path}/cann/set_env.sh
```

## 2. Compiling the Custom Operator Package

The HCCL repository provides a custom operator compilation and packaging project. This project depends on the following files in the repository:

```text
├── build.sh                        # Compilation entry in the hccl repository root directory
├── CMakeLists.txt                  # Compilation and build configuration file in the hccl repository root directory
├── cmake/
|   ├── config.cmake                # CMake variable definitions
|   ├── func.cmake                  # CMake function definitions
|   ├── package.cmake               # Signature and packaging function definitions
|   ├── makeself_custom.cmake       # MakeSelf packaging logic
├── scripts/
    ├── custom/install.sh           # Custom operator package installation script
    └── sign/add_header_sign.py     # AICPU operator package signing script
```

Therefore, developers first need to download the hccl repository, then run `build.sh` from the repository root directory for compilation, specifying the custom operator project path using `custom_ops_path`:

```bash
# Download the hccl repository
git clone https://gitcode.com/cann/hccl.git

# Compile the custom operator package
cd hccl
bash build.sh --vendor=cust --ops=allgather_aicpu --custom_ops_path=./examples/05_custom_ops_allgather/aicpu
```

> Where:
> 
> - `--vendor` specifies the custom operator identifier.
> - `--ops` specifies the custom operator name.
> - `--custom_ops_path` specifies the custom operator project path.

## 3. Installing the Custom Operator Package

The custom operator installation package is located in the `./build_out` directory. Install it using the `--install` parameter:

```bash
./build_out/cann-hccl_custom_allgather_aicpu_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> Where:
> 
> - `<arch>` is the system architecture of the current compilation environment.
> - `<ascend_cann_path>` is an optional parameter indicating the CANN software package installation directory. The default value is the CANN software package path where the `ASCEND_CUSTOM_OPP_PATH` or `ASCEND_OPP_PATH` environment variable is located.

The custom operator package installation information is as follows:

- Header file: `${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_allgather.h`
- Dynamic library: `${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_allgather_aicpu.so`
- AICPU operator description file: `${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/libcustom_allgather_aicpu_kernel.json`
- AICPU operator package: `${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_allgather.tar.gz`
- Installation script: `${ASCEND_HOME_PATH}/opp/vendors/cust/scripts/install.sh`

> `${ASCEND_HOME_PATH}` is the CANN-Toolkit installation path.

## 4. Running the Custom Operator

### 4.1 Disable AICPU Operator Signature Verification

The AICPU operator package `aicpu_hccl_custom_allgather.tar.gz` generated from source compilation is loaded to the Device when the service starts. During loading, the driver performs security signature verification by default to ensure that the package is trustworthy. Because the operator package compiled by developers from source does not contain signature information, disable the driver security signature verification mechanism.

**Method to disable signature verification:**

Use Ascend HDK 25.5.T2.B001 or later, and use the npu-smi tool provided with the Ascend HDK to disable signature verification. The following reference commands must be executed as the root user on the physical machine (using device 0 as an example).

```shell
npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # Enable signature verification configuration
npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # Disable user custom signature verification
```

Where:

- `-i` specifies the device ID, which is the NPU ID obtained from the `npu-smi info -l` command.
- `-d` specifies the attribute value for the corresponding configuration item.

> Note:
> Disabling the driver security signature verification mechanism carries certain security risks. Users must ensure the security and reliability of custom communication operators to prevent malicious attacks.

### 4.2 Modify the AICPU Whitelist

By default, AICPU only loads packages configured in the whitelist. User-developed AICPU operator packages must be added to the whitelist:

```bash
# Edit the configuration file, using the root user default installation path as an example
vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
```

Append the following content to `ascend_package_load.ini`:

```ini
name:aicpu_hccl_custom_allgather.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/aicpu/kernel
load_as_per_soc:false
```

Field descriptions:

- `name`: tar package file name.
- `install_path`: Installation path enumeration value on the Device side. The AICPU kernel file path must be set to 2.
- `optional`: The default value is true. If the corresponding package does not exist, skip loading.
- `package_path`: Relative path of the tar package under the CANN Toolkit package on the Host side.
- `load_as_per_soc`: Whether to load for each chip type.

### 4.3 Compile the Sample

Run the following commands in the `examples/05_custom_ops_allgather/aicpu/testcase` directory:

```bash
# Compile the sample
make
```

### 4.4 Run the Sample

Run the following commands in the `examples/05_custom_ops_allgather/aicpu/testcase` directory:

```bash
# Run the sample
make test

# Or run the sample binary directly
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH}
./custom_allgather_aicpu
```

### 4.5 Sample Output

The input data of all nodes is initialized to the Device ID of that node. After successful execution, the terminal displays log output similar to the following (using 2 cards as an example):

```text
Found 2 NPU device(s) available
rankId: 1, input: [1 1]
rankId: 0, input: [0 0]
rankId: 0, output: [ 0 0 1 1 ]
rankId: 1, output: [ 0 0 1 1 ]
```
