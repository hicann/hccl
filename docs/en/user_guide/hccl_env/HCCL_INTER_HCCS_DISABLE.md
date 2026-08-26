# HCCL_INTER_HCCS_DISABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:55:24.418Z pushedAt=2026-08-06T06:44:10.651Z -->

## Function

This environment variable is used to configure the communication link type within a SuperPoD in SuperPoD networking. The following values are supported:

- `TRUE`: AI nodes within the SuperPoD use RoCE for RDMA communication.

- `FALSE`: AI nodes within the SuperPoD use HCCS communication links for SDMA communication.

The default value is `FALSE`.

## Configuration Example

```bash
export HCCL_INTER_HCCS_DISABLE=FALSE
```

## Constraints

None.

## Applicable Products

Atlas A3 training products/Atlas A3 inference products
