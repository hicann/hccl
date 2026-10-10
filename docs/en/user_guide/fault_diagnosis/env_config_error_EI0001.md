# Environment Variable Configuration Error (EI0001)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:51:12.062Z pushedAt=2026-09-07T08:21:17.648Z -->

## Troubleshooting Approach

The appearance of the EI0001 fault code in service logs indicates an HCCL environment variable configuration error. Generally, the ERROR MESSAGE in the printed logs and the CANN logs display the name of the environment variable with the configuration error, the cause of the error, and the valid configuration range. For any questions, refer to [Environment Variable Reference](../hccl_env/README.md).

## HCCL_RDMA_SL Configuration Error (EI0001)

### Symptoms

The log contains the keyword `EI0001` or `Value *** for environment variable *** is invalid`, as shown below:

```text
[PID:3729526]2025-10-23-17:30:40.098.984Config_Error_Invalid_Environment_Variable(EI0001): Value 1000 for environment variable HCCL_RDMA_SL is invalid. Expected value : range[0, 7].
```

<!-- npu="A3,910b,910,310p" id4 -->
For the following products, the keyword "externalinput.cc" in the ERROR logs of the CANN logs indicates that an error is reported when reading the environment variable configuration.

  <!-- npu="A3" id1 -->
- Atlas A3 Training Products/Atlas A3 Inference Products
  <!-- end id1 -->
  <!-- npu="910b" id2 -->
- Atlas A2 Training Products/Atlas A2 Inference Products
  <!-- end id2 -->
  <!-- npu="910" id5 -->
- Atlas training products
  <!-- end id5 -->
  <!-- npu="310p" id3 -->
- Atlas inference products
  <!-- end id3 -->

An example of the error is as follows:

```text
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.098.973 [externalinput.cc:963] [3729526][Parse][rdmaServerLevel]HCCL_RDMA_SL[1000] is invalid. except: [0, 7]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.058 [externalinput.cc:169] [3729526][InitGroupStage][EnvConfig]errNo[0x0000000005000001] In init env variable param, parse HCCL_RDMA_SL failed. errno[1]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.063 [externalinput.cc:47] [3729526][InitExternalInput]call trace: hcclRet -> 1
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.068 [op_base.cc:866] [3729526][HcclGetRootInfo]call trace: hcclRet -> 1
```
<!-- end id4 -->

### Possible Causes and Solutions

If the environment variable configuration parameter does not meet the requirements, adjust the value range based on the suggestions printed in the log. If questions remain, refer to the corresponding [Environment Variable Reference](../hccl_env/README.md).

## HCCL_SOCKET_IFNAME Configuration Error (EI0001)

### Symptoms

In the CANN log, the keyword "get host ip fail by socket Ifname" is found, as shown below:

```text
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.432 [sal.cc:501] [925892][InitGroupStage][EnvConfig]set ifname to [abc] by HCCL_SOCKET_IFNAME, but not found in the environment, ifnames in the environment is as follows
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.437 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[lo] ip[127.10.0.1%lo]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.441 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[enp] ip[127.10.0.2%enp]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.447 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[docker0] ip[172.17.0.1%docker0]
```

### Root Cause

The Host NIC was specified through the HCCL_SOCKET_IFNAME environment variable, but the corresponding NIC was not found in the current environment. (In a container scenario, a Host NIC available inside the container must be specified.) The error log lists the Host NICs detected in the current environment.

### Solution

Modify the HCCL_SOCKET_IFNAME environment variable and set it to a Host NIC that exists in the environment.
