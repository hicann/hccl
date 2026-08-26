# Impact of Communication Operator Retry on Overall Network Performance

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:17.446Z pushedAt=2026-08-06T07:46:55.170Z -->

Enabling HCCL communication operator retry will make end-to-end network performance closely relevant to model partitioning. This section details the relationship between the retry function and network performance.

## Defining "Key Communicators"

A key communicator is one where performance changes will cause significant changes in the overall network end-to-end performance. This means the communicator is highly important and represents the performance bottleneck of the entire network.

Generally, an entire network has multiple communicators, and among them there is often one key communicator. The performance analysis in this section focuses on this key communicator.

As shown in the following figure:

![Key communicator example](figures/key_comm_domain_example.png)

In the above profiling, there are four communicators where actual communication occurs: Group_777, Group_1289, Group_257, and Group_9.

The BatchSendRecv operator executed in Group_1289 is introduced by Pipeline Parallel. Typically, it is asynchronous communication that can occur not in sync with computation, and its time proportion is not significant, so it is not a key communicator.

Group_777 and Group_9 have fewer operator execution operations and have a relatively small impact on the global scope, so they are also not key communicators.

From this, it can be determined that Group_257 is the key communicator. If the performance of this communicator degrades, it will directly affect the overall network end-to-end performance.

## Relationship Between Overall Network Performance Degradation and Key Communicator

- Focus 1: Whether retry is enabled for the key communicator.

    In some common deployment modes, such as tensor parallelism (TP) combined with data parallelism (DP), TP is the key communicator. If the TP scope is within a server (TP ≤ 16), retry is not enabled for communication operators within the server, so the end-to-end performance is not affected.

    Non-key communicators, on the other hand, have minimal impact on the overall network performance. For example, the following data is from a lab-tested model:

    | Model | Partitioning | Degradation | Description |
    | --- | --- | --- | --- |
    | Llama3-8B<br>(running on a 64-die cluster) | TP=16 (key communicator)<br>DP=4 | 0.03% | Retry is enabled only for the non-critical communicator DP, with minimal impact on end-to-end performance. |
    | GPT4_dropLess<br>(running on a 128-die cluster) | TP=8 (key communicator)<br>PP=1<br>EP=1<br>CP=16 | 0.99% | Retry is enabled only for the non-key communicator CP (context parallelism), with minimal impact on end-to-end performance. |
    | Qwen3-moe-235B (running on a 128-die cluster) | TP=8 (key communicator)<br>PP=1<br>EP=64 | -0.1% | Retry is enabled only for the non-key communicator EP (expert parallelism), with minimal impact on end-to-end performance. |

- Focus 2: Whether the communication operator expansion and computation in the key communicator can overlap.

    If retry is enabled for a key communicator, the performance of that communicator will inevitably degrade. However, whether this degradation triggers overall network degradation depends on whether the operator expansion to AI CPU of that key communicator can overlap with computation.

    After retry is enabled for a single communicator, the most significant difference is that asynchronous operator expansion changes to synchronous one, as shown in the following figure.

    ![Example of operator expansion mode change](figures/expand_mode_change.png)

    Whether the communication operator expansion time can be hidden by computation is the key factor determining whether the communicator affects end-to-end performance. This needs to be analyzed in conjunction with the specific computation operator conditions (model structure).

    As shown in the following figure, the computation operator takes only 50 us. Since the AI_CPU unrolling mode introduces a 150 us gap between the preceding and following communication operators, "150 - 50 = 100 us" is the overhead introduced by retry. If this overhead falls on a key communicator, it will cause end-to-end degradation.

    ![Key communicator performance degradation example](figures/key_comm_domain_degrade.png)

    However, the exact degradation ratio depends on the proportion of the key communicator operators in the overall network (which is strongly correlated with the model structure and deployment approach), as well as whether the operator expansion can overlap with computation.

    For example, even with the same EP=64 partitioning, different models exhibit different degradation results.

    | Model | Partitioning | Degradation | Description |
    | --- | --- | --- | --- |
    | DeepSeekV3 (running on a 64-die cluster) | EP=64 | 0.06% | Retry is enabled on the key communicator EP, but the model has a long computation time, so the retry overhead can be masked by computation. The overall network end-to-end performance degradation is not prominent. |
    | qwen3-moe-30b<br>(running on a 64-die cluster) | EP=64 | 3% | Retry is enabled on the key communicator EP, and the retry overhead cannot be masked by computation. The overall network end-to-end performance degrades.<br>Note: The key communicator EP has poor device affinity when spanning across SuperPoDs. Enabling retry further degrades the overall network performance. |

    Therefore, the factors affecting model end-to-end performance are strongly correlated with the model structure, and the performance impact of retry on the overall network performance must be evaluated based on actual conditions.
    