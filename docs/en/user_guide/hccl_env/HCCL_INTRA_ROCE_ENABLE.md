# HCCL_INTRA_ROCE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:55:57.498Z pushedAt=2026-08-06T06:44:14.005Z -->

## Function

Configures whether to use RoCE links for intra-server or intra-SuperPoD communication.

- For <!-- npu="910" id1 -->Atlas training products and <!-- end id1 -->Atlas A2 training products/Atlas A2 inference products, this environment variable configures whether to use RoCE links for intra-server communication and defaults to `0`. It can be set independently or used together with `HCCL_INTRA_PCIE_ENABLE`. The supported configuration combinations and the intra-server communication links used under different combinations are shown in the following table:

   Supported combinations of HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE

  | HCCL_INTRA_PCIE_ENABLE | HCCL_INTRA_ROCE_ENABLE | Intra-server Communication Link |
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
    > - HCCL_INTRA_PCIE_ENABLE must be set when HCCL_INTRA_ROCE_ENABLE is 0.

- For Atlas A3 training products/Atlas A3 inference products, this environment variable takes effect only when LLM-DataDist is used as the cluster management component. It specifies whether to use RoCE links for intra-SuperPoD communication and defaults to `0`. Its value options are:

  - `0`: The default HCCS links or PCIe links are used for intra-SuperPoD communication (including both LLM-DataDist communication and HCCL communication).

  - `1`: For Atlas 800T A3, Atlas 800I A3, and Atlas 900 A3, RoCE links are used for intra-SuperPoD LLM-DataDist communication, not for HCCL communication. For A200T A3 Box8, RoCE links are used for both LLM-DataDist and HCCL communication.

## Configuration Example

```bash
export HCCL_INTRA_ROCE_ENABLE=1
```

## Constraints

[Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) has two modules, left and right, with devices 0 to 7 and devices 8 to 15 respectively. For this product:

**In single-server use cases**, when PCIe links are used for communication within a server, if devices from both modules need to be used simultaneously, the two modules must use the same number of devices and be in the same plane. That is, device 0 and device 8, device 1 and device 9 (and so on) must be used together. When RoCE links are used for communication within a server, this restriction does not apply.

## Applicable Products

Atlas A3 training products/Atlas A3 inference products: Valid only when LLM-DataDist is used as the cluster management component.

Atlas A2 training products/Atlas A2 inference products: [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) only

<!-- npu="910" id2 -->

Atlas training products: [Atlas 300T Pro](https://support.huawei.com/enterprise/en/ascend-computing/atlas-300t-pro-pid-256118195) only

<!-- end id2 -->