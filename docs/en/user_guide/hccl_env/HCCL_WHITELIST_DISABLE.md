# HCCL_WHITELIST_DISABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:57:27.695Z pushedAt=2026-08-10T09:00:35.760Z -->

## Function

Whether to enable the communication trustlist when using HCCL.

- `0`: Yes. Only IP addresses in the trustlist are allowed to perform collective communication.

- `1`: No. The trustlist is not verified.

The default value is `1`. If trustlist verification is enabled, specify the trustist configuration file path through [HCCL_WHITELIST_FILE](HCCL_WHITELIST_FILE.md).

## Configuration Example

```bash
export HCCL_WHITELIST_DISABLE=1
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