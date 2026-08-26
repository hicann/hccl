# HCCL_SOCKET_IFNAME

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:57:35.811Z pushedAt=2026-08-10T08:57:06.416Z -->

## Function

Configures the name of the communication NIC used by the host during HCCL initialization. HCCL obtains the host IP through this NIC name and communicates with the root node to complete communicator creation.

You can choose one of the following rules for configuration:

- `eth`: Uses all NICs with the prefix `eth`.

    If multiple NIC prefixes are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME=eth,enp` indicates that all NICs with the prefix `eth` or `enp` are used.

- `^eth`: Does not use NICs with the prefix `eth`.

    If multiple NIC prefixes are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME=^eth,enp` indicates that no NICs with the prefix `eth` or `enp` are used.

- `=eth0`: Uses the eth0 NIC.

    If multiple NICs are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME==eth0,enp0` indicates using the eth0 NIC or the enp0 NIC.

- `^=eth0`: Does not use the eth0 NIC.

    If multiple NICs are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME=^=eth0,enp0` indicates not using the eth0 and enp0 NICs.

> [!NOTE] Note
>
> - Multiple NICs can be configured in HCCL_SOCKET_IFNAME, and the first matched NIC is used for communication.
> - [HCCL_IF_IP](HCCL_IF_IP.md) takes precedence over HCCL_SOCKET_IFNAME.
> - If you do not specify HCCL_IF_IP or HCCL_SOCKET_IFNAME, the following priorities are used for selection:
>    NICs other than docker/lo (in ascending lexicographical order of NIC names) \> docker NIC \> lo NIC
>
> If HCCL_IF_IP or HCCL_SOCKET_IFNAME is not specified, the system automatically selects a NIC based on the priority. If the NIC selected on the current node cannot communicate with the NIC selected on the root node, HCCL link establishment will fail.

## Configuration Example

```bash
# Use the eth0 or endvnic NIC.
export HCCL_SOCKET_IFNAME==eth0,endvnic
```

## Constraints

None.

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products

<!-- npu="910" id1 -->

Atlas training products

<!-- end id1 -->

<!-- npu="310p" id2 -->

Atlas inference products

<!-- end id2 -->