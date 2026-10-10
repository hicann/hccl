# HCCL_INTRA_PCIE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:37:59.139Z pushedAt=2026-09-18T08:03:58.203Z -->

## Description

Configures whether to use PCIe links for intra-server communication.

This environment variable defaults to `1`. It can be set independently or used together with `HCCL_INTRA_ROCE_ENABLE`. The supported configuration combinations and the communication links used for intra-server communication under different combinations are shown in the following table:

Supported combinations of HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE

| HCCL_INTRA_PCIE_ENABLE | HCCL_INTRA_ROCE_ENABLE | Intra-Server Communication Link |
| --- | --- | --- |
| 1 | Not set | PCIe |
| 1 | 0 | PCIe |
| 0 | 1 | RoCE |
| Not set | 1 | RoCE |
| 0 | 0 | PCIe |
| Not set | Not set | PCIe |

> [!NOTE]
>
> - HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE can't both be 1.
> - HCCL_INTRA_PCIE_ENABLE can't be 0 when HCCL_INTRA_ROCE_ENABLE is not set.
> - HCCL_INTRA_ROCE_ENABLE can't be 0 when HCCL_INTRA_PCIE_ENABLE is not set.

## Configuration Example

```bash
export HCCL_INTRA_PCIE_ENABLE=1
```

<!-- npu="910b" id3 -->
## Constraints

[Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) has two modules, left and right, with devices 0 to 7 and devices 8 to 15 respectively. For this product:

**In single-node use cases**, when PCIe links are used for intra-server communication, if devices from both modules need to be used simultaneously, the two modules must use the same number of devices and be in the same plane, that is, device 0 and device 8, device 1 and device 9 (and so on) must be used together. When RoCE links are used for intra-server communication, this restriction does not apply.
<!-- end id3 -->

## Applicable Products

<!-- npu="950" id4 -->
- 950PR/950DT: Not supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id5 -->
<!-- npu="910b" id2 -->
- Atlas A2 training products/Atlas A2 inference products: [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) only
<!-- end id2 -->
<!-- npu="910" id1 -->
- Atlas training products: [Atlas 300T Pro](https://support.huawei.com/enterprise/en/ascend-computing/atlas-300t-pro-pid-256118195) only
<!-- end id1 -->
<!-- npu="310p" id6 -->
- Atlas inference products: Not supported
<!-- end id6 -->
