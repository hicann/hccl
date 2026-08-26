# HCCL_HOST_SOCKET_PORT_RANGE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:55:19.364Z pushedAt=2026-08-06T06:44:05.517Z -->

## Function

When a communicator is created based on root node information, use this environment variable to configure the communication ports used by HCCL on the host.

This environment variable can be set to a specific port, a port range, or the string `auto`.

- If a specific port number or port range is specified, the planned number of ports should be no less than the number of HCCL processes on a single NPU. The port number ranges from 1 to 65535. Ensure that the specified ports are not occupied by other processes. Note that ports 1 to 1023 are system reserved ports. Avoid using these ports.

    Specific port numbers and port ranges can be used in combination, separated by commas (,). However, port numbers or port ranges between commas must not overlap. For details, see [Configuration Example](#configuration-example).

- If set to `auto`, the host communication port used by HCCL is dynamically allocated by the operating system.

## Configuration Example

```bash
# Method 1: Configure as a port range.
export HCCL_HOST_SOCKET_PORT_RANGE="60000-60050"
# Method 2: Use specific port numbers together with a port range, separated by commas (,).
export HCCL_HOST_SOCKET_PORT_RANGE="60000,60050-60100,60150-60160"
# Method 3: Specify specific port numbers, separated by commas (,).
export HCCL_HOST_SOCKET_PORT_RANGE="56000,56005,56007,56008,56100,56105,56107,56108"
# Method 4: The operating system dynamically allocates port numbers.
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
```

## Constraints

- In single-device multi-process use cases (where multiple processes share one NPU), configure this environment variable. Otherwise, services may fail due to port conflicts. Note that multi-process running incurs resource overheads and affects communication performance.

- This environment variable takes precedence over [HCCL_IF_BASE_PORT](HCCL_IF_BASE_PORT.md). Setting this environment variable means it decides the communication ports used by HCCL on the host.

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products
