# HCCL_RDMA_QP_PORT_CONFIG_PATH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:56:34.292Z pushedAt=2026-08-10T07:59:42.578Z -->

## Function

During RDMA communication between two ranks, one QP (Queue Pair) is created by default for data transfer. If you want RDMA communication between two ranks to use multiple QPs and specify the source port numbers for multi-QP communication, you can use this environment variable.

Specifically, you can specify the storage path of the configuration file that defines the mapping between `<srcIP,dstIP>` and port numbers. When multiple port numbers are configured for `<srcIP,dstIP>`, the system enables multi-QP communication, and the configured port numbers serve as the source port for each QP.

The following is an example of configuring this environment variable:

```bash
export HCCL_RDMA_QP_PORT_CONFIG_PATH=/home/tmp
```

Here, `/home/tmp` is the storage path of the configuration file `MultiQpSrcPort.cfg` that maps `<srcIP,dstIP>` and ports. Both absolute and relative paths are supported, and the maximum path length must be less than or equal to 4,096 characters.

The `MultiQpSrcPort.cfg` file is custom (note that the file name must remain "MultiQpSrcPort.cfg"). The configuration format is as follows:

```text
srcIP1,dstIP1=srcPort0,srcPort1,...,srcPortN
srcIPN,dstIPN=srcPort0,srcPort1,...,srcPortN
```

- The maximum number of lines allowed in this file is 131072 (128 x 1024).

- Each `<srcIP,dstIP>` address pair supports a maximum of 32 ports, but 8 or fewer ports are recommended. When the number of QPs exceeds 8, performance gains cannot be guaranteed, and excess memory use may cause service execution failures.

- Each `<srcIP,dstIP>` address pair is allowed to appear only once in this file.

- srcIP and dstIP must be in standard IPv4 or IPv6 format.

- srcIP and dstIP can be configured as `0.0.0.0`, which represents all IP addresses.

The following is a configuration example of the `MultiQpSrcPort.cfg` file:

```text
192.168.100.2,192.168.100.3=61100,61101,61102
192.168.100.4,192.168.100.5=61100,61101,61102,61104
0.0.0.0,192.168.100.122=65515,65516,65513
```

## Configuration Example

```bash
export HCCL_RDMA_QP_PORT_CONFIG_PATH=/home/tmp
```

## Constraints

- This environment variable supports only single-operator calls and does not support static graphs.

- This environment variable has a higher priority than [HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md). After this environment variable is configured, the number of QPs used for communication between two ranks is determined by the number of source port numbers configured in the `MultiQpSrcPort.cfg` file.

- The priorities of QP-related configurations are as follows:

    Management-plane multi-QP configuration (configured via the "-s multi_qp" parameter of hccn_tool) \> NSLB QP configuration (configured via the "-t nslb-dp" parameter of hccn_tool) \> Environment variable HCCL_RDMA_QP_PORT_CONFIG_PATH \> Environment variable HCCL_RDMA_QPS_PER_CONNECTION

## Applicable Products

Atlas A2 training products/Atlas A2 inference products

Atlas A3 training products/Atlas A3 inference products
