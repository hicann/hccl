# HCOMM_TA_CTP_UB_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:36:16.240Z pushedAt=2026-09-18T08:03:58.254Z -->

## Description

Configures the jetty timeout coefficient for the UB_CTP protocol.

For 950PR/950DT, this environment variable is an integer ranging from \[0, 31\] and defaults to `8`.

Under the UB_CTP protocol, the jetty timeout is divided into 4 levels. These levels are calculated as `timeout` divided by 8 (even division), where `timeout` is the value of this environment variable. Level 0: 512 ms; Level 1: 4s; Level 2: 8s; Level 3: 32s. The UB_CTP protocol uses the configured value of this environment variable directly, without comparing it with the total TP timeout. It is advised to set the value to 0/8/16/24.

## Configuration Example

```bash
# If the timeout coefficient of the UB_CTP protocol is set to 8, the timeout level is: 8 / 8 = 1, corresponding to 4s.
export HCOMM_TA_CTP_UB_TIMEOUT=8
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id1 -->
- 950PR/950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
