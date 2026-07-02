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
|   ├── hccl_custom_allgather.h         # Custom operator external interface header file
|   ├── common.h                        # Common type definitions and macros
|   ├── aiv_all_gather_mesh_1d.h        # AIV AllGather core algorithm implementation
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

This sample provides a CMake-based build process and a Makefile-based test execution script.

### 2.1 Compile the Custom Operator Library

Run the following commands in the sample root directory:

```bash
# 1. Create a build directory
mkdir build

# 2. Enter the build directory
cd build

# 3. Run CMake configuration
cmake ..

# 4. Compile the project (generates libhccl_custom_allgather.so)
make
```

### 2.2 Run Test Cases

After compilation, enter the `testcase` directory to run the tests:

```bash
# 5. Enter the test case directory
cd ../testcase

# 6. Compile and run the test case
# This command automatically compiles the test program, sets LD_LIBRARY_PATH, and runs using mpirun
make run
```

### 2.3 Expected Results

After successful execution, the terminal displays log output similar to the following (using 2 cards as an example):

```text
[INFO] MPI Initialized. World Size: 2
[INFO] Device 0 selected (Total devices: 8)
[INFO] Device 1 selected (Total devices: 8)
[INFO] HCCL Comm Initialized
[INFO] Buffers allocated and initialized
[INFO] Starting HcclAllGatherCustom...
[INFO] HcclAllGatherCustom completed and synchronized
[INFO] Test Passed!
```
