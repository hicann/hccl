# HCCL_INTRA_PCIE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:55:45.848Z pushedAt=2026-08-06T06:44:12.163Z -->

## Function

Configures whether to use PCIe links for intra-server communication.

This environment variable defaults to `1`. It can be set independently or used together with `HCCL_INTRA_ROCE_ENABLE`. The supported configuration combinations and the communication links used for intra-server communication under different combinations are shown in the following table:

Supported combinations of HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE

| HCCL_INTRA_PCIE_ENABLE | HCCL_INTRA_ROCE_ENABLE | Intra-server Communication Link |
| --- | --- | --- |
| 1 | Not set | PCIe |
| 1 | 0 | PCIe |
| 0 | 1 | RoCE |
| Not set | 1 | RoCE |
| 0 | 0 | PCIe |
| Not set | Not set | PCIe |

> [!NOTE] Note
>
> - HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE can't both be 1.
> - HCCL_INTRA_PCIE_ENABLE can't be 0 when HCCL_INTRA_ROCE_ENABLE is not set.
> - HCCL_INTRA_ROCE_ENABLE can't be 0 when HCCL_INTRA_PCIE_ENABLE is not set.

## Configuration Example

```bash
export HCCL_INTRA_PCIE_ENABLE=1
```

## Constraints

[Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) has two modules, left and right, with devices 0 to 7 and devices 8 to 15 respectively. For this product:

**In single-node use cases**, when PCIe links are used for intra-server communication, if devices from both modules need to be used simultaneously, the two modules must use the same number of devices and be in the same plane, that is, device 0 and device 8, device 1 and device 9 (and so on) must be used together. When RoCE links are used for intra-server communication, this restriction does not apply.

## Applicable Products

<!-- npu="910" id1 -->

Atlas training products: [Atlas 300T Pro](https://support.huawei.com/enterprise/en/ascend-computing/atlas-300t-pro-pid-256118195) only

<!-- end id1 -->

Atlas A2 training products/Atlas A2 inference products: [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) only
