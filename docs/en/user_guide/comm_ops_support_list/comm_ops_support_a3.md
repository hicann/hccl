# Atlas A3 Training Products/Atlas A3 Inference Products

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:57:28.416Z pushedAt=2026-08-17T02:49:43.071Z -->

This section provides the communication operator support status for the Atlas A3 training products/Atlas A3 inference products.

- Single-Operator Zero-Copy: To reduce memory copy overhead, HCCL can directly operate on the memory passed by the service, thereby improving communication performance.

- Communication Operator Re-execution: When a network failure causes a communication interruption, HCCL attempts to re-execute the communication operator, thereby improving communication stability.

- Deterministic Computation: Under the same hardware and input conditions, reduction-type communication operators produce the same output across multiple executions.

> [!NOTE]Note
>
> - The following presents the support status of communication operators based on their expansion modes. Expansion modes not listed are not supported.
> - In the tables of this section, "✓" indicates supported, "×" indicates not supported, and "NA" indicates not applicable.
> - Operators and network running modes not listed are not supported.

## AI CPU

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Single-Operator Zero-Copy</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Re-execution</p></th>
<th><p>Intra-node Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
<th><p>Inter-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>✓</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>✓</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="2"><p>Scatter</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td rowspan="2"><p>BatchSendRecv</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
</tr>
</tbody>
</table>

## AIV

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Single-Operator Zero-Copy</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Re-execution</p></th>
<th><p>Intra-node Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
<th><p>Inter-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single Operator Mode</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>✓</p></td>
<td><p>✓</p></td>
<td><p>×</p></td>
</tr>
</tbody>
</table>

> [!NOTE]Note
> AIV mode delivers better performance for small-size data communication and is primarily used in inference scenarios. In this mode:
>
> - Single-Operator Zero-Copy introduces runtime memory negotiation, which increases communication latency. Therefore, Single-Operator Zero-Copy is not supported in AIV mode.
> - The re-execution feature increases execution time, so operator re-execution is not supported in AIV mode.
> - Only intra-SuperPoD communication is supported; inter-SuperPoD communication is not supported.
