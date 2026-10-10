# HCCL_SOCKET_FAMILY

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:20:03.128Z pushedAt=2026-09-18T08:03:58.239Z -->

## Description

Specifies the IP protocol used by the communication NIC. The following two options are supported:

- `AF_INET`: Uses the IPv4 protocol.
- `AF_INET6`: Uses the IPv6 protocol.

**Defaults to using the IPv4 protocol.**

This environment variable has the following two use cases:

- Specifies the IP protocol version used by the host-side communication NIC during HCCL initialization.

  In this case, this environment variable must be used together with [HCCL_SOCKET_IFNAME](HCCL_SOCKET_IFNAME.md). When HCCL obtains the host IP by specifying an NIC name, this environment variable specifies the socket communication IP protocol of the NIC.

- Specifies the IP protocol version used by the device-side communication NIC during HCCL initialization.

  In this case, if the IP protocol specified by this environment variable does not match the actual NIC information obtained, the actual NIC information in the environment prevails.

  For example, if this environment variable is set to IPv6 but only IPv4 NICs exist on the device, the IPv4 NICs will be used.

**For 950PR/950DT**: This environment variable does not support configuring the IP protocol version for the communication NIC on the device. The device-side NICs on these models use:

- IPv6 for socket link establishment when the UB protocol is used for communication.
- IPv4 for socket link establishment when the UBoE protocol is used for communication.

## Configuration Example

```bash
export HCCL_SOCKET_FAMILY=AF_INET       #IPv4
export HCCL_SOCKET_FAMILY=AF_INET6      #IPv6
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
