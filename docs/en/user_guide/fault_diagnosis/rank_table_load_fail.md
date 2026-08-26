# Rank Table File Load Failure

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:58:30.529Z pushedAt=2026-08-13T06:34:10.081Z -->

## Troubleshooting Approach

The method of creating a communicator based on a rank table requires loading the rank table file. If the file path does not exist, permissions are insufficient, or the file format or configuration is incorrect, HCCL fails to load and returns an error.

The following sections describe some common cases of rank table file validation failure errors. If no matching case is found, troubleshooting can be performed based on the actual error message.

## Rank Table File Read Failure (EI0004)

### Symptom

The CANN log contains the keyword "is not a valid real path", as shown below:

```text
[ERROR] HCCL(1104629,test_one_side):2025-10-28-17:45:13.679.684 [param_check.cc:66] [1104629][InitGroupStage][RanktableConfig]errNo[0x0000000005010001] path /ranktable.json is not a valid real path
```

### Possible Causes

When a communicator is initialized based on a rank table file, the specified rank table file path does not exist or has insufficient permissions.

### Solution

Correct the rank table file path or configure the correct read permission.

## Rank Table Field Configuration Error (EI0004)

### Symptom

For Atlas A3 training series products/Atlas A3 inference series products, the keyword "RanktableCheck" appears in the CANN log, as shown below:

```text
[ERROR] HCCL(1265,):2025-10-21 07:56:47.198.454 [topoinfo_ranktableConcise.cc:727][15326][InitGroupStage][RanktableCheck]errNo[0x0000000005010001] super_device_id[] is invalid
```

### Possible Causes

The "version" field in the rank table is "1.2", but the "super_device_id" field in the rank table is left empty, causing the rank table verification to fail.

### Solution

Add the "super_device_id" field in the rank table file. For configuration instructions, see [Configuring Rank Table Resource Information (Atlas A3 training products/Atlas A3 inference products)](../cluster_info_config/rank_table_config_a3.md).

## Rank Table File device_ip Field Verification Failure (EI0014)

### Symptom

The CANN log contains the keyword `the IP address(***) in the ranktable is inconsistent with the IP(***)address of the network adapter`, as shown below:

```text
[ERROR] HCCP(166192,eExecutor):2025-01-21-16:59:39.962.565 [ra_host.c:480]tid:167056,ra_rdev_init_check_ip(480) : [check][ip]fail, ret(129) the IP address(127.10.0.0) in the ranktable is inconsistent with the IP address(127.10.0.1) of the network adapter, please make sure they're consistent. num(2)
```

### Possible Causes

During device IP verification, HCCL detects that the device IP obtained on the current device side is inconsistent with the device IP configured for the current rank in the rank table, causing the verification to fail.

For example, on rank0, the device IP corresponding to the bound device is 127.10.0.1, but the device IP configured for rank0 in the rank table is 127.10.0.0, which causes the HCCL verification to fail.

### Solution

Check whether the device IP configured in the rank table is consistent with the device IP used by each rank in the communicator.
