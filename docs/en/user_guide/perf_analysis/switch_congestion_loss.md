# Switch Traffic Congestion, Backpressure, or Packet Loss and Retransmission

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T11:08:25.335Z pushedAt=2026-09-18T08:03:58.296Z -->

> [!NOTE]
> The use cases and handling methods described in this section apply only to the following products:
>
> - Atlas A3 training products/Atlas A3 inference products
> - Atlas A2 training products/Atlas A2 inference products

If a notify wait task lasting about 4 seconds is observed in the profile data, it typically indicates a network configuration issue that has caused packet loss and retransmission. You can locate the issue by checking the `roce_new_pkt_rty_num` field in the statistics using the hccn_tool.

If the value of this field increases during task execution, it indicates that packets were lost and re-transmitted on the network. In this case, further troubleshoot the switch configuration.

Run the following command to view the statistics:

```bash
hccn_tool -i {DeviceId} -stat -g
```
