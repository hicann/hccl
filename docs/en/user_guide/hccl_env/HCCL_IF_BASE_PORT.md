# HCCL_IF_BASE_PORT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:55:33.871Z pushedAt=2026-08-06T06:44:07.453Z -->

## Function

When the communicator is created based on root node information, use this environment variable to specify the starting port number of the host NIC. After configuration, the system occupies 32 ports starting from this port for cluster information collection by default.

This environment variable must be configured as an integer in the range of [1024, 65520]. Ensure that the ports to allocate are not occupied.

## Configuration Example

```bash
export HCCL_IF_BASE_PORT=50000
```

## Constraints

In distributed use cases, HCCL uses some ports on the host for cluster information collection. The operating system must reserve these ports.

- If the port is not specified through HCCL_IF_BASE_PORT, HCCL uses ports 60000–60031 by default. Run the following command to reserve this range of operating system ports:

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=60000-60031
    ```

- If the port is specified through HCCL_IF_BASE_PORT, for example, port 50000, HCCL uses ports 50000–50031. Run the following command to reserve this range of operating system ports:

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=50000-50031
    ```

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