# Ascend 950PR/Ascend 950DT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:58:43.081Z pushedAt=2026-08-11T08:55:32.507Z -->

This section provides the communication operator support status for Ascend 950PR/Ascend 950DT.

> [!NOTE]Note
> 
> - The communication operator support status is presented below by expansion mode.
> - In the tables in this section, "√" indicates supported, "×" indicates not supported, and "NA" indicates not applicable.

## AICPU_TS (AI CPU)

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computing</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-POD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
</tbody>
</table>

## AIV

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computing</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-POD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
</tbody>
</table>

## CCU MS Mode

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computing</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-POD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
</tbody>
</table>

## CCU SCHED Mode

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computing</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-POD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-Operator Mode</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Mode Ascend IR</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph Capture Mode aclgraph</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
</tbody>
</table>
