# HCCL_HOST_RDMA_UDP_PORTS_LIST

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:28:10.943Z pushedAt=2026-09-18T08:03:58.190Z -->

## Description

For RoCE communication over host NICs, to specify the UDP source ports used by multiple QPs, configure the UDP source port list by NPU physical device ID using this environment variable.

Set this environment variable in the following format:

```text
<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>[;<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>]
```

- `phy_dev_id` is the NPU physical device ID, set to a non-negative decimal integer. Each NPU physical device ID can be set only once.
- `src_port` is a UDP source port, set to a decimal integer. The value range is \[1, 65535\]. Ports \[1, 1023\] are system-reserved ports. Avoid using them. A maximum of 32 UDP source ports can be set for each NPU physical device ID.
- Separate configurations of multiple NPU physical devices by semicolons (`;`), and multiple UDP source ports of the same NPU physical device by commas (`,`). Spaces are not allowed in the configuration.
- The value of this environment variable cannot exceed 32\*1024 characters.

HCCL selects the corresponding UDP source port list based on the NPU physical device ID currently in use. If the number of QPs is greater than the number of UDP source ports, the ports are used cyclically starting from the first port in the list; if the number of QPs is smaller than the number of UDP source ports, only the first several ports matching the number of QPs are used.

This environment variable is used only to specify UDP source ports and does not change the number of QPs between two ranks. The number of QPs can be configured using the [HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md) environment variable. By default, this environment variable is not configured, and the system selects UDP source ports based on other configurations or the default setting.

## Configuration Example

```bash
export HCCL_HOST_RDMA_UDP_PORTS_LIST="0:10000,10015;1:10016,10031"
```

The preceding configuration indicates that NPU physical device 0 uses UDP source ports 10000 and 10015, and NPU physical device 1 uses UDP source ports 10016 and 10031.

## Constraints

- This environment variable takes effect only in RoCE communication over host NICs.
- If the NPU physical device ID currently in use is not configured in the environment variable, this environment variable does not specify a UDP source port.
- This environment variable takes precedence over [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md). If this environment variable is configured, the UDP source port list specified by this environment variable is used; otherwise, the ports in the configuration file specified by `HCCL_RDMA_QP_PORT_CONFIG_PATH` are used.
- If the configuration format or value is invalid, communicator initialization fails.

## Applicable Products

<!-- npu="950" id1 -->
- 950PR/950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
