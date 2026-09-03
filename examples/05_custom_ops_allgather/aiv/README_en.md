# Custom Communication Operator - AllGather Communication

## Sample Description

This sample demonstrates how to develop an AllGather custom communication operator based on the HCCL AIV communication programming interface. Key features:

1. Implement an AllGather collective communication operator based on the AIV (AI Vector) communication engine.
2. Contains both Host-side operator logic and Device-side Kernel implementation.
3. Provides a complete compilation, build, and test verification process.

## Directory Structure

```text
├── CMakeLists.txt                      # Root directory compilation and build configuration file
├── op_host/
|   ├── CMakeLists.txt
|   ├── all_gather.cc                   # HcclAllGatherCustom operator Host-side implementation
|   ├── launch_kernel.cc                # Kernel submission logic implementation
|   └── launch_kernel.h                 # Kernel submission interface definition
├── op_kernel/
|   ├── CMakeLists.txt
|   └── launch_kernel_asc.asc           # Operator Kernel-side implementation (Ascend C)
├── inc/
    ├── hccl_custom_allgather.h         # Custom operator external interface header file
    ├── common.h                        # Common type definitions and macros
    ├── aiv_all_gather_mesh_1d.h        # AIV AllGather core algorithm implementation
    ├── aiv_communication_base_v2.h     # AIV communication base class
    ├── log.h                           # Logging utility
    ├── extra_args.h                    # Additional parameter definitions
    └── sync_interface.h                # Synchronization interface definition
```

## 1. Environment Preparation

### 1.1 Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2):

- Ascend 950PR/Ascend 950DT
- Atlas A3 training products/Atlas A3 inference products (Only supports intra-super-node communication scenarios)
- Atlas A2 training products/Atlas A2 inference products (Only supports single-device communication scenarios)

### 1.2 Install the CANN Toolkit Development Kit Package

Refer to the [Ascend Documentation Center - CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard) to install the latest version of the CANN Toolkit development kit package.

### 1.3 Configure Environment Variables

Using the root user default installation path as an example:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

Additionally, running test cases requires an MPI environment. Ensure that MPI is installed and configured.

## 2. Compilation and Execution

This sample provides a CMake-based build process.

### 2.1 Compile the Custom Operator Library

Run the following commands in the sample root directory:

```bash
bash build.sh --vendor=cust --ops=allgather_aiv --custom_ops_path=./examples/05_custom_ops_allgather/aiv
```

> Where:
> 
> - `--vendor` specifies the custom operator identifier.
> - `--ops` specifies the custom operator name.
> - `--custom_ops_path` specifies the custom operator project path.

### 2.2 Installing the Custom Operator Package

The custom operator installation package is located in the `./build_out` directory. Install it using the `--install` parameter:

```bash
./build_out/cann-hccl_custom_allgather_aiv_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> Where:
> 
> - `<arch>` is the system architecture of the current compilation environment.
> - `<ascend_cann_path>` is an optional parameter indicating the CANN software package installation directory. The default value is the CANN software package path where the `ASCEND_CUSTOM_OPP_PATH` or `ASCEND_OPP_PATH` environment variable is located.

The custom operator package installation information is as follows:

- Header file: `${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_allgather.h`
- Dynamic library: `${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_allgather.so`

> `${ASCEND_HOME_PATH}` is the CANN-Toolkit installation path.

### 2.3 Run Test Cases

Test Sample has generated at `2.1 Compile the Custom Operator Library`, the binary file path is:
`build/examples/05_custom_ops_allgather/testcase/custom_allgather_test`

```bash
# run the sample binary directly
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH}
cd build/examples/05_custom_ops_allgather/testcase
mpirun -n rank_size ./custom_allgather_test data_len
Parameter Description:
rank_size: used rank number
data_len: date length
```

### 2.4 Expected Results

After successful execution, the terminal displays log output similar to the following (using 2 cards as an example):

```text
[1787902520.136766] [Rank 1] MPI Initialized. World Size: 2
[1787902520.136768] [Rank 0] MPI Initialized. World Size: 2
[1787902520.145917] [Rank 0] Device 0 selected (Total devices: 8)
[1787902520.145918] [Rank 1] Device 1 selected (Total devices: 8)
[1787902520.724696] [Rank 0] Root info generated
[1787902520.724744] [Rank 0] HCCL set device[0]
[1787902520.727436] [Rank 1] HCCL set device[1]
[1787902522.982323] [Rank 0] HCCL Comm Initialized
[1787902522.982908] [Rank 0] Buffers allocated and initialized
[1787902523.008164] [Rank 1] HCCL Comm Initialized
[1787902523.008742] [Rank 1] Buffers allocated and initialized
rank1 dataLen=32 time=439 ms
[1787902523.447898] [Rank 1] VerifyResult Passed!
rank0 dataLen=32 time=465 ms
[1787902523.447966] [Rank 0] VerifyResult Passed!
```
