# Ascend 950PR/Ascend 950DT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-01T03:52:05.289Z pushedAt=2026-09-01T07:06:58.733Z -->

This section describes the support communication operators for Ascend 950PR/Ascend 950DT.

> [!NOTE]
> 
> - The support communication operators are presented by expansion mode.
> - In the following tables, "√" indicates supported, "×" indicates not supported, and "NA" indicates not applicable.

## AICPU_TS (AI CPU)

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-PoD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
<th><p>Inter-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
</tr>
</tbody>
</table>

## AIV

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-PoD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
<th><p>Inter-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-operator</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
</tbody>
</table>

## CCU MS Mode

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-PoD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
<th><p>Inter-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>x</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>x</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>x</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-operator</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>x</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
</tbody>
</table>

## CCU SCHED mode

<table><thead align="left"><tr><th><p>Operator</p></th>
<th><p>Network Running Mode</p></th>
<th><p>Deterministic Computation</p></th>
<th><p>Intra-Chassis Communication</p></th>
<th><p>Intra-PoD Communication</p></th>
<th><p>Intra-SuperPoD Communication</p></th>
<th><p>Inter-SuperPoD Communication</p></th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGather</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Reduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Scatter</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p></td>
<td><p>Single-operator</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p></td>
<td><p>Single-operator</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>√</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Send</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>Recv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p></td>
<td><p>Single-operator</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph (Ascend IR)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
<tr><td><p>Graph capture (ACL graph)</p></td>
<td><p>NA</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
<td><p>×</p></td>
</tr>
</tbody>
</table>
