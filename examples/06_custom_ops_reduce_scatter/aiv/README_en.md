# Custom Communication Operator - ReduceScatter Communication

## Sample Description

This sample demonstrates how to develop an ReduceScatter custom communication operator based on the HCCL AIV communication programming interface. Key features:

1. Implement an ReduceScatter collective communication operator based on the AIV (AI Vector) communication engine.
2. Contains both Host-side operator logic and Device-side Kernel implementation.
3. Provides a complete compilation, build, and test verification process.

## Directory Structure

```text
├── CMakeLists.txt                      # Root directory compilation and build configuration file
├── op_host/
|   ├── CMakeLists.txt
|   ├── reduce_scatter.cc               # HcclReduceScatterCustom operator Host-side implementation
|   ├── launch_kernel.cc                # Kernel submission logic implementation
|   └── launch_kernel.h                 # Kernel submission interface definition
├── op_kernel/
|   ├── CMakeLists.txt
|   └── launch_kernel_asc.asc           # Operator Kernel-side implementation (Ascend C)
├── inc/
|   ├── hccl_custom_reduce_scatter.h    # Custom operator external interface header file
|   ├── common.h                        # Common type definitions and macros
|   ├── aiv_reduce_scatter_mesh_1d.h    # AIV ReduceScatter core algorithm implementation
|   ├── aiv_communication_base_v2.h     # AIV communication base class
|   ├── log.h                           # Logging utility
|   ├── extra_args.h                    # Additional parameter definitions
|   └── sync_interface.h                # Synchronization interface definition
└── testcase/
    ├── CMakeLists.txt                  # Test case CMake configuration file
    ├── Makefile                        # Test case Makefile (for compilation and running)
    └── main.cc                         # Test case main program
```

## 1. Environment Preparation

### 1.1 Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2):

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

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
bash build.sh --vendor=cust --ops=reduce_scatter_aiv --custom_ops_path=./examples/06_custom_ops_reduce_scatter/aiv
```

> Where:
> 
> - `--vendor` specifies the custom operator identifier.
> - `--ops` specifies the custom operator name.
> - `--custom_ops_path` specifies the custom operator project path.

### 2.2 Installing the Custom Operator Package

The custom operator installation package is located in the `./build_out` directory. Install it using the `--install` parameter:

```bash
./build_out/cann-hccl_custom_reduce_scatter_aiv_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> Where:
> 
> - `<arch>` is the system architecture of the current compilation environment.
> - `<ascend_cann_path>` is an optional parameter indicating the CANN software package installation directory. The default value is the CANN software package path where the `ASCEND_CUSTOM_OPP_PATH` or `ASCEND_OPP_PATH` environment variable is located.

The custom operator package installation information is as follows:

- Header file: `${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_reduce_scatter.h`
- Dynamic library: `${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_reduce_scatter.so`

> `${ASCEND_HOME_PATH}` is the CANN-Toolkit installation path.

### 2.3 Run Test Cases

Test Sample has generated at `2.1 Compile the Custom Operator Library`, the binary file path is:
`./build/examples/06_custom_ops_reduce_scatter/testcase/custom_reduce_scatter_test`

# run the sample binary directly
mpirun -n rank_size ./build/examples/06_custom_ops_reduce_scatter/testcase/custom_reduce_scatter_test data_len
Parameter Description:
rank_size: used rank number
data_len: date length
```

### 2.3 Expected Results

After successful execution, the terminal displays log output similar to the following (using 2 cards as an example):

```text
[1786071476.120968] [Rank 0] MPI Initialized. World Size: 2
[1786071476.120968] [Rank 1] MPI Initialized. World Size: 2
[1786071476.127411] [Rank 0] Device 0 selected (Total devices: 8)
[1786071476.127411] [Rank 1] Device 1 selected (Total devices: 8)
[1786071478.023709] [Rank 0] Root info generated
[1786071478.023786] [Rank 0] HCCL set device[0]
[1786071478.023778] [Rank 1] HCCL set device[1]
[1786071483.214938] [Rank 0] HCCL Comm Initialized
[1786071483.221873] [Rank 0] Buffers allocated and initialized
[1786071483.254098] [Rank 1] HCCL Comm Initialized
[1786071483.259378] [Rank 1] Buffers allocated and initialized
rank1 dataLen=1024 time=835 ms
[1786071484.095144] [Rank 1] VerifyResult Passed!
rank0 dataLen=1024 time=873 ms
[1786071484.095200] [Rank 0] VerifyResult Passed!
```
