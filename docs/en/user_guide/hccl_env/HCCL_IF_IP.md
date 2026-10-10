# HCCL_IF_IP

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:34:06.246Z pushedAt=2026-09-18T08:03:58.198Z -->

## Function

When the communicator is created based on root node information, use this environment variable to configure the communication IP address used by the host during HCCL initialization. This IP address is used to communicate with the root node to complete communicator creation.

The value is a string, which must be in standard IPv4 or IPv6 format. Currently, only host NICs are supported, and only one IP address can be set.

HCCL selects the host communication NIC in the following priority order:

HCCL_IF_IP \> [HCCL_SOCKET_IFNAME](HCCL_SOCKET_IFNAME.md) \> NICs other than docker/lo (in ascending lexicographic order of NIC names) \> docker NIC \> lo NIC.

> [!NOTE]
> If HCCL_IF_IP or HCCL_SOCKET_IFNAME is not configured, the system automatically selects a NIC based on the priority order. If the NIC selected for the current node cannot communicate with the NIC selected for the root node, HCCL link establishment will fail.

## Configuration Example

```bash
export HCCL_IF_IP=10.10.10.1
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id3 -->
- 950PR/950DT: Supported
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas inference products: Supported
<!-- end id2 -->
