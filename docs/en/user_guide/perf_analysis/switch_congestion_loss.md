# Switch Traffic Congestion, Backpressure, or Packet Loss and Retransmission

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-10T09:34:06.746Z pushedAt=2026-08-10T12:04:44.798Z -->

If a notify wait task lasting about 4 seconds is observed in the profile data, it typically indicates a network configuration issue that has caused packet loss and retransmission. You can locate the issue by checking the `roce_new_pkt_rty_num` field in the statistics using the hccn_tool.

If the value of this field increases during task execution, it indicates that packet loss and retransmission has occurred on the network. In this case, further troubleshoot the switch configuration.

Run the following command to view the statistics:

```bash
hccn_tool -i {DeviceId} -stat -g
```
