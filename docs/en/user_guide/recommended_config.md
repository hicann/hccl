# Recommended Business Configuration

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T08:10:30.900Z pushedAt=2026-09-08T03:27:51.442Z -->

This section provides recommended business configurations for common business scenarios of the following products:

  <!-- npu="950" id5 -->
- Ascend 950PR/Ascend 950DT
  <!-- end id5 -->
  <!-- npu="A3" id3 -->
- Atlas A3 Training Products/Atlas A3 Inference Products
  <!-- end id3 -->
  <!-- npu="910b" id4 -->
- Atlas A2 Training Products/Atlas A2 Inference Products
  <!-- end id4 -->

> [!NOTE]Note
> This section only provides functional descriptions and configuration examples of the recommended configuration environment variables. For detailed usage instructions, see [Environment Variable Reference](./hccl_env/README.md).

<!-- npu="950" id6 -->
## Ascend 950PR/Ascend 950DT

- **Training Scenario**

  | Environment Variable | Configuration Description |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | Configures the socket link establishment timeout waiting time. The default value is 120, in seconds. In this scenario, it is recommended to adjust the link establishment timeout waiting time based on the network scale.<br>export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, it is recommended to keep the default value "AICPU_TS", which represents that the communication operator is expanded on the AI CPU.<br>export HCCL_OP_EXPANSION_MODE="AI_CPU" |

- **Inference Scenario**

    | Environment Variable | Configuration Instructions |
    | --- | --- |
    | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, it is recommended to set it to "AIV", indicating that communication operators are expanded on Vector Core.<br>export HCCL_OP_EXPANSION_MODE="AIV" |

- **Reinforcement Learning Training-Inference Integration**

  | Environment Variable | Configuration Description |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | Configures the socket link establishment timeout waiting time. The default value is 120, in seconds. In this scenario, it is recommended to adjust the link establishment timeout waiting time based on the network scale.<br>export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, it is recommended to keep the default value "AICPU_TS", which represents that the communication operator is expanded on the AI CPU.<br>export HCCL_OP_EXPANSION_MODE="AI_CPU"<br>Note:<br>For the inference communicator, the operator expansion location of the inference communicator must be set to "Vector Core" through communicator-level configuration parameters. For PyTorch framework networks, this can be configured through the "hccl_op_expansion_mode" parameter. The configuration method is as follows:<br>options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()<br>   options.hccl_config ={"hccl_op_expansion_mode":3}<br>   torch.distributed.init_process_group(backend="hccl", pg_options=options)<br>For detailed introduction of PyTorch framework parameters, search for "Configuring HCCL Communicator Parameters Through pg_options" in the *[TorchNPU Product Documentation](https://hiascend.com/en/document/redirect/pytorchuserguide)*. |
<!-- end id6 -->

<!-- npu="A3" id1 -->
## Atlas A3 Training Products/Atlas A3 Inference Products

- **Training Scenario**

  | Environment Variable | Configuration Instructions |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | Configures the socket link establishment timeout wait time. Default value: 120, in seconds. In this scenario, adjust the link establishment timeout wait time based on the network scale.<br>export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, keep the default value "AI_CPU", which means communication operators are expanded on the AI CPU.<br>export HCCL_OP_EXPANSION_MODE="AI_CPU" |

- **Inference Scenario**
  - Prefill-Decode Hybrid Deployment

    | Environment Variable | Configuration Instructions |
    | --- | --- |
    | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, it is recommended to set it to "AIV", indicating that communication operators are expanded on Vector Core.<br>export HCCL_OP_EXPANSION_MODE="AIV" |
    | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | Whether to enable deterministic computing. Users can enable or disable it based on the usage scenario. Default value: false, which represents disabled deterministic computing.<br>export HCCL_DETERMINISTIC=false |

  - Prefill-Decode Disaggregated Deployment

    | Environment Variable | Configuration Instructions |
    | --- | --- |
    | [HCCL_INTRA_ROCE_ENABLE](./hccl_env/HCCL_INTRA_ROCE_ENABLE.md) | In scenarios where only LLM-DataDist is used as the cluster management component, it is recommended to configure intra-SuperPoD communication using RoCE links through this environment variable. In non-LLM-DataDist scenarios, no configuration is required.<br>export HCCL_INTRA_ROCE_ENABLE=1 |
    | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, it is recommended to set it to "AIV", indicating that communication operators are expanded on Vector Core.<br>export HCCL_OP_EXPANSION_MODE="AIV" |
    | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | Whether to enable deterministic computing. Users can enable or disable it based on the usage scenario. Default value: false, which represents disabled deterministic computing.<br>export HCCL_DETERMINISTIC=false |

- **Reinforcement Learning Training-Inference Integration**

  | Environment Variable | Configuration Description |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | Configures the socket link establishment timeout waiting time. The default value is 120, in seconds. In this scenario, it is recommended to adjust the link establishment timeout waiting time based on the network scale.<br>export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, it is recommended to keep the default value "AI_CPU", which represents that the communication operator is expanded on the AI CPU.<br>export HCCL_OP_EXPANSION_MODE="AI_CPU"<br>Note:<br>For the inference communicator, the operator expansion location of the inference communicator must be set to "Vector Core" through communicator-level configuration parameters. For PyTorch framework networks, this can be configured through the "hccl_op_expansion_mode" parameter. The configuration method is as follows:<br>options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()<br>   options.hccl_config ={"hccl_op_expansion_mode":3}<br>   torch.distributed.init_process_group(backend="hccl", pg_options=options)<br>For detailed introduction of PyTorch framework parameters, search for "Configuring HCCL Communicator Parameters Through pg_options" in the *[TorchNPU Product Documentation](https://hiascend.com/en/document/redirect/pytorchuserguide)*. |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | Configures whether to enable deterministic computing. Users can enable or disable it based on the usage scenario. The default value is false, which represents that deterministic computing is disabled.<br>export HCCL_DETERMINISTIC=false |
<!-- end id1 -->

<!-- npu="910b" id2 -->
## Atlas A2 Training Products/Atlas A2 Inference Products

- **Training Scenario**

  | Environment Variable | Configuration Description |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | Configures the socket link establishment timeout waiting time. The default value is 120, in seconds. In this scenario, it is recommended to adjust the link establishment timeout waiting time based on the network scale.export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the expansion mode of communication operators.<br>In this scenario, it is recommended to keep the default value "HOST", which represents that the communication operator is expanded on the host-side CPU.<br>export HCCL_OP_EXPANSION_MODE="HOST" |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | Configures whether to enable deterministic computing. Users can enable or disable it based on the usage scenario. The default value is false, which represents that deterministic computing is disabled.<br>export HCCL_DETERMINISTIC=false |

- **Inference Scenario**

  | Environment Variable | Configuration Instructions |
  | --- | --- |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the communication operator expansion mode.<br>In this scenario, keep the default value "HOST", which represents communication operator expansion on the Host-side CPU.<br>export HCCL_OP_EXPANSION_MODE="HOST" |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | Whether to enable deterministic computing. Users can enable or disable it based on the usage scenario. Default value: false, which represents disabled deterministic computing.<br>export HCCL_DETERMINISTIC=false |

- **Reinforcement Learning Training-Inference Integration**

  | Environment Variable | Configuration Instructions |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | Configures the socket link establishment timeout wait time. Default value: 120, in seconds. In this scenario, adjust the link establishment timeout wait time based on the network scale.<br>export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | Configures the communication operator expansion mode.<br>In this scenario, keep the default value "HOST", which represents communication operator expansion on the Host-side CPU.<br>export HCCL_OP_EXPANSION_MODE="HOST" |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | Whether to enable deterministic computing. Users can enable or disable it based on the usage scenario. Default value: false, which represents disabled deterministic computing.<br>export HCCL_DETERMINISTIC=false |
<!-- end id2 -->
