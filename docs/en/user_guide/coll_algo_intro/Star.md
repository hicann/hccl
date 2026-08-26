# Star

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:54:37.181Z pushedAt=2026-08-04T10:26:18.654Z -->

## Algorithm Description

The Star algorithm is applicable to rooted communication operations (such as Broadcast, Reduce, Gather, and Scatter), and uses a star topology or fully connected topology to complete the communication operation in a single step. Taking the Broadcast operator as an example, the Star algorithm implementation is shown in the following figure. The root node uses the star topology to collect data from other nodes.

![](figures/star_algo_principle.png)

## Time Consumption Calculation

If the communication data size between each non-root node and the root node is defined as n, the communication time consumption of the entire Star algorithm is: $\alpha +\beta n$.
