# HCCL_WHITELIST_FILE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:29:16.097Z pushedAt=2026-09-18T08:03:58.249Z -->

## Description

When communication trustlist verification is enabled through HCCL_WHITELIST_DISABLE, use this environment variable to specify the path to the HCCL communication trustlist configuration file. Only IP addresses in the communication trustlist are allowed to perform collective communication.

The format of the HCCL communication trustlist configuration file is:

```text
{ "host_ip": ["ip1", "ip2"], "device_ip": ["ip1", "ip2"] } 
```

Where:

- `device_ip` is a reserved field and is not supported in the current version.
- The IP address format is dotted decimal notation.

> [!NOTE]
> The trustlist IP must be specified as a valid IP used for cluster communication.

## Configuration Example

```bash
export HCCL_WHITELIST_FILE=/home/test/whitelist
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id3 -->
- 950PR/950DT: Supported
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas inference products: Supported
<!-- end id2 -->
