# Atlas A2 Training Products/Atlas A2 Inference Products

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:45:45.438Z pushedAt=2026-09-07T07:40:23.247Z -->

This section provides the communication operator support status for the Atlas A2 training products/Atlas A2 inference products.

- Single-Operator Zero-Copy: To reduce memory copy overhead, HCCL can directly operate on the memory passed by the service, thereby improving communication performance.
- Communication Operator Re-execution: When a network failure causes a communication interruption, HCCL attempts to re-execute the communication operator, thereby improving communication stability.
- Deterministic Computation: Under the same hardware and input conditions, reduction-type communication operators produce the same output across multiple executions.

> [!NOTE]Note
>
> - The communication operator support status is presented below by expansion mode. Expansion modes not listed are not supported.
> - In the tables in this section, "√" indicates supported, "×" indicates not supported, and "NA" indicates not applicable. The Atlas A2 training products/Atlas A2 inference products do not support Single-Operator Zero-Copy or Re-execution.
> - Operators and network running modes not listed are not supported.

## HOST/HOST_TS

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Single-Operator Zero-Copy</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Re-execution</p></th>
<th><p>Intra-Node Communication</p></th>
<th><p>Inter-Node Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
</tbody>
</table>

## AIV

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Single-Operator Zero-Copy</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Re-execution</p></th>
<th><p>Intra-Node Communication</p></th>
<th><p>Inter-Node Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
</tbody>
</table>
