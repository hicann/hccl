# Cluster Information Verification Failure Issues

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:57:43.653Z pushedAt=2026-08-13T06:37:55.223Z -->

## Fault Locating Approach

HCCL verifies the rank table file or the rank table information collected through negotiation. If the verification fails, HCCL reports an error and exits. Locate the fault based on the actual error message.

Possible causes include: rank table file verification failure, content mismatch with the hardware configuration, inconsistent TLS configuration, or duplicate superDeviceId.

The following sections present common cases of cluster information verification failure errors. If no matching case is found, locate and troubleshoot based on the actual error message.

## IP Family Verification Inconsistency (EI0001)

### Symptom

The CANN log contains the keyword "rank\[\*\] device ip family\[2\] is not same as others\[\*\]." as shown below:

```text
[ERROR] HCCL(144905,python):2025-04-20-00:26:54.435.048 [config.cc:413] [145735][InitGroupStage][RanktableCheck]rank[0] device ip family[2] is not same as others[10].
```

### Possible Causes

The two ranks obtain different IP families. For example, one side uses IPv4 while the other uses IPv6.

### Solution

Query whether IPv4 is configured:

```bash
hccn_tool -i {deviceId} -ip -g
```

Query whether IPv6 is configured:

```bash
hccn_tool -i {deviceId} -ip -inet6 -g
```

All ranks in the same job must use the same IP Family. By default, HCCL uses the IPv4 protocol first. If no IP address with the IPv4 protocol is configured on the device side, the IP address corresponding to the IPv6 protocol is used instead. You can use the HCCL_SOCKET_FAMILY environment variable to specify the NIC IP protocol to be used.

**Note**: The family value is printed as an enumeration value. The enumeration values and their corresponding relationships are shown in the following table.

| IP Family Enumeration Value | IP Protocol |
| --- | --- |
| 2 | IPv4 |
| 10 | IPv6 |

## TLS Information Configuration Inconsistency (EI0016)

### Symptom

The CANN log contains the keyword "All ranks are consistent.", as shown below:

```text
[ERROR] HCCL(94774,all_reduce_test):2025-10-27-11:51:32.570.490 [topoinfo_exchange_agent.cc:831] [94774][InitGroupStage][RanktableCheck] Value Disable for config "tls" is invalid. Expected Value:"All ranks are consistent. Current status : rankList for enabled tls:[10.78.106.107/0]; rankList for disabled tls:[10.78.106.107/0]; rankList for query failure tls:".;
```

### Possible Causes

During communicator creation, after the server node receives information about all ranks in the communicator, it verifies whether the TLS configurations of all ranks in the communicator are consistent. If any configuration inconsistency is detected, the verification fails immediately and the process exits. A list of nodes with Disable or Enable is printed, while the nodes not listed have the opposite TLS configuration.

This verification feature is supported only in Ascend HDK 25.2.0 or later and only when the communicator is initialized through root information negotiation. Ascend 950PR/Ascend 950DT does not support this feature.

### Solution

1. Query the TLS status switch on each server involved in collective communication.

    Run the following command on the server to obtain the TLS switch status.

    ```bash
    hccn_tool -i <device_id> -tls -g
    ```

    <device_id\> is the logical ID of the device. You can also use the following for loop to query the TLS information of all devices at once.

    ```bash
    for i in `seq 0 7`; do hccn_tool -i $i -tls -g; done    # 0 and 7 are the start and end values of the device IDs to be queried, respectively.
    ```

The printed information is as follows:

    ```text
    dev_id:0, tls switch[0](0:disable, 1:enable), tls alarm time threshold[60]days
    dev_id:0, [pub cert] info:
             issuer[/C=CN/ST=GD/O=HUAWEI/OU=2012/CN=2_1thCA]
             start_time[Wed Feb 19 03:19:21 2020 GMT]
             end_time[Sat Feb 16 03:19:21 2030 GMT]
    dev_id:0, [ca1 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:07 2020 GMT]
             end_time[Sat Feb 16 03:19:07 2030 GMT]
    dev_id:0, [ca2 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:10 2020 GMT]
             end_time[Sat Feb 16 03:19:10 2030 GMT]
    dev_id:1, tls switch[0](0:disable, 1:enable), tls alarm time threshold[60]days
    dev_id:1, [pub cert] info:
             issuer[/C=CN/ST=GD/O=HUAWEI/OU=2012/CN=2_1thCA]
             start_time[Wed Feb 19 03:19:21 2020 GMT]
             end_time[Sat Feb 16 03:19:21 2030 GMT]
    dev_id:1, [ca1 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:07 2020 GMT]
             end_time[Sat Feb 16 03:19:07 2030 GMT]
    dev_id:1, [ca2 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:10 2020 GMT]
             end_time[Sat Feb 16 03:19:10 2030 GMT]
    ... ...
    ```

    In the output, `tls switch[0]` indicates that the TLS status is disabled, and `switch[1]` indicates that the TLS status is enabled.

2. Determine whether the TLS status switches of all devices on each server are consistent.

    - If they are inconsistent, it is recommended to uniformly set the TLS status to enabled. If the TLS switch is disabled, information may be subject to eavesdropping, tampering, and impersonation during collective communication.

        Use the following command to modify the TLS status switch:

        ```bash
        hccn_tool -i <device_id> -tls -s enable 1
        ```

        `enable` is the status switch. Setting it to `1` indicates enabled, and setting it to `0` indicates disabled.

    - If they are consistent and the status is enabled, proceed to step 3 to determine whether the TLS certificate information of each node is consistent.

3. Check whether the TLS certificate information of each device on all servers is consistent.

    You can determine whether the TLS certificate information of each device is consistent based on the information obtained in step 1. If it is inconsistent, you can replace the certificate suite by running the following command.

    ```bash
    hccn_tool -i 0 -tls -s path /root pri pri.pem pub pub.pem ca1 ca1.pem ca2 ca2.pem crl xxx.crl
    ```

    `-i` specifies the device ID. `-s path` specifies the storage path for the certificate, private key, and certificate revocation list. `pri` specifies the private key file name. `pub` specifies the device certificate file name. `ca1`, `ca2`, and `crl` specify the root certificate, secondary root certificate, and certificate revocation list file names, respectively.

    For more usage and parameter descriptions of the hccn_tool, see the corresponding version of *[HCCN Tool Interface Reference](https://support.huawei.com/enterprise/en/ascend-computing/ascend-hdk-pid-252764743?category=developer-documents&subcategory=interface-reference)*.

## Duplicate superDeviceId (EI0014)

### Symptom

The CANN log contains the keyword "superDeviceId\[\*\*\*\] in superPod\[\*\*\*\]is already exist", as shown below:

```text
[ERROR] HCCL(169030,alltoall_test):2025-10-23-16:28:59.392.635 [topoinfo_exchange_agent.cc:695] [169030][InitGroupStage][RanktableCheck]devices have same superDeviceId[0x3000000] in superPod[super_pod_id_0]. Current device info: serverId[127.10.0.1], rankId[0], group[hccl_world_group]. Another device info: rankId[1].
```

### Possible Causes

The superDeviceId is the physical ID of a Device within an Atlas A3 training product or Atlas A3 inference product in a SuperPoD system, serving as the unique identifier of the Device in the SuperPoD system. During the consistency check, HCCL detected duplicate superDeviceId values within a SuperPoD, causing the check to fail. The superDeviceId can be queried using the npu-smi command:

```bash
npu-smi info -t spod-info -i id -c chip_id
```

- id: Device ID. The NPU ID obtained through the `npu-smi info -l` command is the device ID.

- chip_id: Chip ID. The Chip ID obtained through the `npu-smi info -m` command is the chip ID.

The "SDID" field in the command output is the superDeviceId.

Possible causes of this issue:

- Abnormal hardware configuration.

- Different physical SuperPoDs are configured in the same logical SuperPoD through the [HCCL_LOGIC_SUPERPOD_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md) environment variable, causing duplicate superDeviceId.

### Solution

Modify the hardware configuration or correctly configure the [HCCL_LOGIC_SUPERPOD_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md) environment variable to prevent devices with the same superDeviceId from appearing within the same superpod.
