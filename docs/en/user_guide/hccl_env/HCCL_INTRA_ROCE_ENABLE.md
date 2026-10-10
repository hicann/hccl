# HCCL_INTRA_ROCE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:40:11.659Z pushedAt=2026-09-18T08:03:58.205Z -->

## Description

Configures whether to use RoCE links for intra-server or intra-SuperPoD communication.

  <!-- npu="910b,910" id5 -->
- For Atlas training products and Atlas A2 training products/Atlas A2 inference products, this environment variable configures whether to use RoCE links for intra-server communication and defaults to `0`. It can be set independently or used together with `HCCL_INTRA_PCIE_ENABLE`. The supported configuration combinations and the intra-server communication links used under different combinations are shown in the following table:

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
    > - HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE can't both be 1.
    > - HCCL_INTRA_PCIE_ENABLE can't be 0 when HCCL_INTRA_ROCE_ENABLE is not set.
    > - HCCL_INTRA_PCIE_ENABLE can't be unspecified when HCCL_INTRA_ROCE_ENABLE is 0.
  <!-- end id5 -->

  <!-- npu="A3" id1 -->
- For Atlas A3 training products/Atlas A3 inference products, this environment variable takes effect only when LLM-DataDist is used as the cluster management component. It specifies whether to use RoCE links for intra-SuperPoD communication and defaults to `0`. Its value options are:
  - `0`: The default HCCS links or PCIe links are used for intra-SuperPoD communication (including both LLM-DataDist communication and HCCL communication).
  - `1`: For Atlas 800T A3, Atlas 800I A3, and Atlas 900 A3, RoCE links are used for intra-SuperPoD LLM-DataDist communication, while HCCL communication is not impacted. For A200T A3 Box8, RoCE links are used for both LLM-DataDist and HCCL communication.
  <!-- end id1 -->

## Configuration Example

```bash
export HCCL_INTRA_ROCE_ENABLE=1
```

<!-- npu="910b" id6 -->
## Constraints

[Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) has two modules, left and right, with devices 0 to 7 and devices 8 to 15 respectively. For this product:

**In single-node use cases**, when PCIe links are used for intra-server communication, if devices from both modules need to be used simultaneously, the two modules must use the same number of devices and be in the same plane, that is, device 0 and device 8, device 1 and device 9 (and so on) must be used together. When RoCE links are used for intra-server communication, this restriction does not apply.
<!-- end id6 -->

## Applicable Products

<!-- npu="950" id7 -->
- 950PR/950DT: Not supported
<!-- end id7 -->
<!-- npu="A3" id3 -->
- Atlas A3 training products/Atlas A3 inference products: Valid only when LLM-DataDist is used as the cluster management component.
<!-- end id3 -->
<!-- npu="910b" id4 -->
- Atlas A2 training products/Atlas A2 inference products: [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) only
<!-- end id4 -->
<!-- npu="910" id2 -->
- Atlas training products: [Atlas 300T Pro](https://support.huawei.com/enterprise/en/ascend-computing/atlas-300t-pro-pid-256118195) only
<!-- end id2 -->
<!-- npu="310p" id8 -->
- Atlas inference products: Not supported
<!-- end id8 -->
