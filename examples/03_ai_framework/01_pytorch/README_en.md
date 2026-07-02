# Performing AllReduce Operations Using PyTorch

## Sample Description

This sample demonstrates how to perform an AllReduce operation using the PyTorch interface. It covers the following features:

- Device detection: Query the number of available devices using the `torch_npu.npu.device_count()` interface.
- Start multiple processes using the `torch.multiprocessing.spawn()` interface.
- In each process, initialize the communication domain using the `torch.distributed.init_process_group()` interface.
- In each process, perform the AllReduce operation using the `torch.distributed.all_reduce()` interface.

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

## Running the Sample

```bash
python hccl_pytorch_allreduce_test.py
```

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the expansion mode of communication operators. For the supported ranges for different product models, see the usage instructions for this environment variable in the [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the communication operator expansion mode to the AI CPU communication engine
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Sample Output

The data of each rank is initialized to 0 through 7. After the AllReduce operation, the result on each rank is the sum of the data at the corresponding positions of all ranks (the data of 8 ranks is added).

```
[Rank 0] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:0')
[Rank 1] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:1')
[Rank 2] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:2')
[Rank 3] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:3')
[Rank 4] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:4')
[Rank 5] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:5')
[Rank 6] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:6')
[Rank 7] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:7')

[Rank 0] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:0')
[Rank 1] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:1')
[Rank 2] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:2')
[Rank 3] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:3')
[Rank 4] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:4')
[Rank 5] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:5')
[Rank 6] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:6')
[Rank 7] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:7')
```
