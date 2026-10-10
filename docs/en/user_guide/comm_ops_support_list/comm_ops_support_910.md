# Atlas Training Products

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:44:37.693Z pushedAt=2026-09-11T02:41:23.300Z -->

This section provides the communication operator support status for the Atlas training products.

- Single-Operator Zero-Copy: To reduce memory copy overhead, HCCL can directly operate on the memory passed by the service, thereby improving communication performance.
- Communication Operator Re-execution: When a network failure causes a communication interruption, HCCL attempts to re-execute the communication operator, thereby improving communication stability.
- Deterministic Computation: Under the same hardware and input conditions, reduction-type communication operators produce the same output across multiple executions.

> [!NOTE]Note
>
> - For Atlas training products, communication operators support only HOST expansion.
> - In the tables in this section, "√" indicates supported, "×" indicates not supported, and "NA" indicates not applicable. Atlas training products do not support single-operator zero-copy or re-execution.
> - Operators and network operation modes not listed are not supported.

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Single-Operator Zero-Copy</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Re-execution</p></th>
<th><p>Intra-Node Communication</p></th>
<th><p>Inter-Node Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="2"><p>Broadcast</p></td>
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
<tr><td rowspan="2"><p>AllGather</p></td>
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
<tr><td rowspan="2"><p>Reduce</p></td>
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
<tr><td rowspan="2"><p>AllReduce</p></td>
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
<tr><td rowspan="2"><p>Scatter</p></td>
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
<tr><td rowspan="2"><p>ReduceScatter</p></td>
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
<tr><td rowspan="2"><p>AlltoAll</p></td>
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
<tr><td rowspan="2"><p>AlltoAllV</p></td>
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
<tr><td rowspan="2"><p>Send</p></td>
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
<tr><td rowspan="2"><p>Recv</p></td>
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
<tr><td rowspan="2"><p>BatchSendRecv</p></td>
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
</tbody>
</table>
