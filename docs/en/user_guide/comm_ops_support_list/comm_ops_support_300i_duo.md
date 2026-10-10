# Atlas Inference Products

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:44:33.464Z pushedAt=2026-09-11T02:04:52.429Z -->

This section provides the communication operator support status for Atlas inference products.

- Single-Operator Zero-Copy: To reduce memory copy overhead, HCCL can directly operate on the memory passed by the service, thereby improving communication performance.
- Communication Operator Re-execution: When a network failure causes a communication interruption, HCCL attempts to re-execute the communication operator, thereby improving communication stability.
- Deterministic Computation: Under the same hardware and input conditions, reduction-type communication operators produce the same output across multiple executions.

> [!NOTE]Note
>
> - The following presents the support status of communication operators by their expansion modes. Expansion modes not listed are not supported.
> - In the tables in this section, "√" indicates supported, "×" indicates not supported, and "NA" indicates not applicable. Atlas inference products do not support single-operator zero copy or re-execution.
> - Operators and network running modes not listed are not supported.

## HOST

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Single Operator Zero-Copy</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Re-execution</p></th>
<th><p>Intra-node Communication</p></th>
<th><p>Inter-node Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>AllGather</p></td>
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
<td><p>×</p></td>
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
<td><p>×</p></td>
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
<td><p>×</p></td>
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
<tr><td rowspan="3"><p>Send</p></td>
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
<tr><td rowspan="3"><p>Recv</p></td>
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
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
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

## AI CPU

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Single Operator Zero-Copy</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Re-execution</p></th>
<th><p>Intra-node Communication</p></th>
<th><p>Inter-node Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="2"><p>AllReduce</p></td>
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
</tbody>
</table>
