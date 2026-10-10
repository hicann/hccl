# HCOMM_TA_RTP_UB_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:39:46.534Z pushedAt=2026-09-18T08:03:58.257Z -->

## Description

Configures the jetty timeout coefficient under the UB_RTP protocol.

For 950PR/950DT, this environment variable is an integer ranging from \[0, 31\] and defaults to `16`.

Under the UB_RTP protocol, the jetty timeout duration is divided into 4 levels. These levels are calculated as `timeout` divided by 8 (even division), where `timeout` is the value of this environment variable. Level 0: 512 ms; Level 1: 4s; Level 2: 8s; Level 3: 32s. Built-in interception checks apply to this config. Before creating a jetty, the software first checks the TP timeout configuration. If the time configured by this environment variable is less than or equal to the TP total timeout duration, the jetty timeout duration is automatically promoted to the smallest level greater than the TP total timeout duration. If the time configured by this environment variable is greater than the TP total timeout duration, the level configured by this environment variable is used. It is advised to set the value to 0/8/16/24.

## Configuration Example

```bash
# If this variable is set to 16 for the UB_RTP protocol, the timeout duration level is: 16 / 8 = 2, corresponding to 8s.
export HCOMM_TA_RTP_UB_TIMEOUT=16
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
