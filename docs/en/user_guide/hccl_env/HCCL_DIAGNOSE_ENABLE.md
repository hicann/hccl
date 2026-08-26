# HCCL_DIAGNOSE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:45.965Z pushedAt=2026-08-06T06:43:59.669Z -->

## Function

This environment variable is used to configure whether to cache detailed information of some tasks during collective communication, so that when a task fails, detailed logs are printed for issue locating.

The following values are supported:

- `1`: Enables collective communication caching.

- `0`: Disables collective communication caching.

Defaults to `0`.

Note that enabling this environment variable affects performance.

## Configuration Example

```bash
export HCCL_DIAGNOSE_ENABLE=1
```

## Constraints

A maximum of 2,000 latest operator information entries can be saved.

## Applicable Products

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products
