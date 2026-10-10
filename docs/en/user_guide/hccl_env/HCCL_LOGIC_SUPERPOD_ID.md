# HCCL_LOGIC_SUPERPOD_ID

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:42:51.440Z pushedAt=2026-09-18T08:03:58.208Z -->

## Description

For the SuperPoD networking of Atlas A3 training products/Atlas A3 inference products, if you do not use a rank table to configure cluster resource information, you can use this environment variable to specify the SuperPoD ID to which the current node's running process belongs, thereby dividing a physical SuperPoD into multiple logical SuperPoDs.

The value of this environment variable is a string with fewer than 128 characters, and defaults to an empty string.

If this environment variable is not configured, the value of `Super Pod ID` in the environment is obtained as the SuperPoD ID. You can query this value by running `npu-smi info -t spod-info -i <id> -c <chip_id>`.

## Configuration Example

```bash
export HCCL_LOGIC_SUPERPOD_ID=super_pod_id_1
```

## Constraints

- This environment variable applies only when no rank table is used to configure cluster information in SuperPoD networking. If a rank table is used, the configuration in the rank table takes precedence.
- This environment variable is used to divide a physical SuperPoD into multiple logical SuperPoDs. It does not support configuring ranks that belong to different physical SuperPoDs into a single logical SuperPoD.

## Applicable Products

<!-- npu="950" id1 -->
- 950PR/950DT: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
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
