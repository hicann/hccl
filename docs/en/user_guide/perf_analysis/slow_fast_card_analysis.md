# Slow/Fast Device Analysis

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-10T09:34:00.453Z pushedAt=2026-08-10T11:58:52.369Z -->

Before executing memory copy tasks, a collective communication operator needs to perform pre-synchronization with the peer to ensure that the peer is ready to receive data from the local end. Therefore, if the peer has not yet reached the same communication operator, the local end must wait for the peer to get in sync before proceeding. This notify wait time is also counted in the communication operator execution, resulting in slow communication operator performance.

![Slow/fast devices issue](figures/slow_fast_rank.png)

As shown in the preceding figure, device1 reaches the AllReduce operator first, but device0 reaches the AllReduce operator later. As a result, the AllReduce operator on device1 incurs a period of notify wait time that is counted in the operator's execution time. In this case, you need to further investigate the cause of the slow/fast execution of operators across devices. Common causes include:

- Performance fluctuation of preceding computation operators.

- Communication operator dispatch bottleneck, for example, other behaviors on the host cause the communication operator of the slow device to fail to be dispatched in time.
