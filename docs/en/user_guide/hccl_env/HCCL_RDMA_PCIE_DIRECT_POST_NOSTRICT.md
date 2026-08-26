# HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:56:28.483Z pushedAt=2026-08-10T07:49:30.447Z -->

## Function

In multi-node communication where the host OS small-page memory page table size is not 4 KB, when communication operator dispatch performance is host-bound, you can set this environment variable to submit RDMA tasks via PCIe Direct, thereby improving communication operator dispatch performance.

This environment variable supports the following value options:

- `TRUE`: Submits RDMA tasks via PCIe Direct (a high-speed communication interface between the host and device).

- `FALSE` (default): Submits RDMA tasks via HDC (Host Device Communication).

This environment variable takes effect only when the host-side small-page memory page table size is not 4 KB. If that table size is 4 KB, RDMA tasks are submitted using PCIe Direct regardless of the value of this environment variable.

Note:

- When this environment variable is set to `TRUE`, additional large-page memory on the device is consumed (each communication link consumes an extra 1 MB of large-page memory).

- If you want to use this environment variable to improve communication operator dispatch performance while saving device-side large-page memory use, you can set the inter-server communication algorithm to ring by using [HCCL_ALGO](HCCL_ALGO.md) to control the number of communication links.

  ```bash
  export HCCL_ALGO="level0:NA;level1:ring"
  ```

## Configuration Example

```bash
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
```

## Constraints

When using this environment variable, the context described in [Function](#function) must be met, that is:

- Multi-node communication

- The host OS small-page memory page table size is not 4 KB.

## Applicable Products

Atlas A2 training products/Atlas A2 inference products
