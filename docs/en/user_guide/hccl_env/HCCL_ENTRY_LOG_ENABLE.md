# HCCL_ENTRY_LOG_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:51.982Z pushedAt=2026-08-06T06:44:01.610Z -->

## Function

This environment variable controls whether to print the call behavior logs of communication operators in real time.

- `1`: Prints logs in real time. Each time a communication operator is called, one run log is printed.

- `0`: Does not print logs.

Defaults to `0`.

The default run log storage path of HCCL is `$HOME/ascend/log/run/plog/plog-_pid__\*.log`. For details about logs, see [Log Reference](https://hiascend.com/document/redirect/CannCommunitylogref).

## Configuration Example

```bash
export HCCL_ENTRY_LOG_ENABLE=1
```

## Constraints

Only used for single-operator calls of collective communication operators.

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