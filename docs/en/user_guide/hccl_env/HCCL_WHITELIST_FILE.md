# HCCL_WHITELIST_FILE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:57:54.782Z pushedAt=2026-08-10T09:04:35.076Z -->

## Function

When communication trustlist verification is enabled through HCCL_WHITELIST_DISABLE, use this environment variable to specify the path to the HCCL communication trustlist configuration file. Only IP addresses in the communication trustlist are allowed to perform collective communication.

The format of the HCCL communication trustlist configuration file is:

```text
{ "host_ip": ["ip1", "ip2"], "device_ip": ["ip1", "ip2"] } 
```

Where:

- `device_ip` is a reserved field and is not supported in the current version.

- The IP address format is dotted decimal notation.

> [!NOTE] Note
> The trustlist IP must be specified as a valid IP used for cluster communication.

## Configuration Example

```bash
export HCCL_WHITELIST_FILE=/home/test/whitelist
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