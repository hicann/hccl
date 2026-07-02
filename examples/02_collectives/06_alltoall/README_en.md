# Collective Communication - AlltoAll

## Sample Description

This sample demonstrates how to call the `HcclAlltoAll()` API to perform an `AlltoAll` operation. It covers the following functions:

- Call `aclrtGetDeviceCount()`  to detect devices and query the number of available devices.
- Call `HcclGetRootInfo()` and use `rank 0` as the root rank to generate the rootinfo identifier.

  > The rootinfo identifier contains the device IP address and device ID. This information must be broadcast to all ranks in the cluster to initialize the communicator.

- In each thread, call `HcclCommInitRootInfo()` to initialize the communicator based on the rootinfo identifier.
- Call `HcclAlltoAll()` to split the input data into a specific number of blocks along a given dimension, send the blocks sequentially to other ranks, receive data from other ranks, concatenate the received data along the specific dimension in order, and display the result.

## Directory Structure

```
├── main.cc     # Sample source file
├── Makefile    # Compilation and build configuration file
└── alltoall    # Compiled executable file
```

## Environment Preparation

### Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2):

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 Training Series Products</term> / <term>Atlas A3 Inference Series Products</term>
- <term>Atlas A2 Training Series Products</term>
- <term>Atlas Training Series Products</term> / <term>Atlas Inference Series Products</term>

### Setting Environment Variables

```bash
# Set CANN environment variables. The following uses the root user default installation path as an example.
source /usr/local/Ascend/cann/set_env.sh
```

## Compiling and Running the Sample

Run the following commands in the sample code directory:

```bash
make
make test
```

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the expansion mode of communication operators. For the supported ranges for different product models, see the usage instructions for this environment variable in the [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the communication operator expansion mode to the AI CPU communication engine
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Sample Output

The data of each rank is initialized to the corresponding rank ID. After the AlltoAll operation, the content of each node is the concatenation of the input data of all nodes.

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 1, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 2, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 3, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 4, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 5, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 6, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 7, output: [ 0 1 2 3 4 5 6 7 ]
```
