# HCCL_ENTRY_LOG_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:21:48.059Z pushedAt=2026-09-18T08:03:58.185Z -->

## Description

This environment variable controls whether to print the call behavior logs of communication operators in real time.

- `1`: Prints logs in real time. Each time a communication operator is called, one run log is printed.
- `0`: Does not print logs.

Defaults to `0`.

The default run log storage path of HCCL is `$HOME/ascend/log/run/plog/plog-pid_*.log`. For details about logs, see [Log Reference](https://hiascend.com/en/document/redirect/CannCommunitylogref).

## Configuration Example

```bash
export HCCL_ENTRY_LOG_ENABLE=1
```

## Constraints

Only used for single-operator calls of collective communication operators.

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
