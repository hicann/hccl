# Performing AllReduce Operations Using TensorFlow

## Sample Description

This sample demonstrates how to perform an AllReduce operation using the TensorFlow interface. It covers the following functions:

- Initialize the communicator based on the `ranktable.json` configuration file.

## Environment Preparation

### Environment Requirements

This sample supports the following products in a single-server 8-card configuration:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training products/Atlas A3 inference products
- Atlas A2 training products
- Atlas training products
- Atlas inference products

Note: This sample code is developed based on TensorFlow 1.x and is not compatible with TensorFlow 2.x. TensorFlow 1.15.0 is recommended.

### Setting Environment Variables

```bash
# Set CANN environment variables. The following uses the root user default installation path as an example.
source /usr/local/Ascend/cann/set_env.sh

# Set the path to the rank_table.json configuration file
export RANK_TABLE_FILE=ranktable.json
```

## Running the Sample

```bash
bash run_tensorflow.sh
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
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:0 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:1 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:2 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:3 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:4 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:5 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:6 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:7 tensorflow hccl test success
```
