# RFC: Bandwidth-efficient Invariant ReduceScatter (BIRS) Algorithm

- Start Date: 2026-04-24
- RFC PR: cann/hccl#657
- Related Issues: cann/hcomm#139, cann/hccl#96

---

## Summary

Introduce BIRS (Batchsize Invariant ReduceScatter) — a novel batch-invariant ReduceScatter algorithm for the Ascend A3 server topology in HCCL. While guaranteeing deterministic reduction ordering (bit-level reproducibility), the algorithm achieves up to 25% performance improvement (operator execution time without submission overhead) over the existing RHD (Recursive Halving-Doubling) algorithm for large message sizes by more fully utilizing the SIO + HCCS hybrid interconnect bandwidth.

## Background and Motivation

### Industry Demand for Deterministic Collective Communication

In distributed training and inference, **deterministic collective communication** requires that reduction operations (AllReduce, ReduceScatter, etc.) produce **bit-identical** results for the same input, regardless of batch size, process count, or memory sharding strategy. This requirement has become a hard constraint in multiple industry scenarios:

#### 1. Training Reproducibility and CI/CD

Reproducible training is essential for trustworthy research and production pipelines. Non‑deterministic reductions introduce floating‑point noise that masks bugs and makes results impossible to compare across runs.

- **Picard (2021)** ("Torch.manual_seed(3407) is all you need") demonstrates that random seed variations alone can produce statistically significant outliers in final model performance – when reduction ordering is also non‑deterministic, the variance grows even larger. ([arXiv:2109.08203](https://arxiv.org/abs/2109.08203))
- **CI/CD and Debugging**: In continuous integration testing and distributed debugging, any non‑determinism turns a reproducible bug into a ghost. Deterministic collectives guarantee that a failing test will fail identically on every rerun, drastically reducing root‑cause analysis time.

#### 2. Reinforcement Learning (RL, RLHF, PPO)

Reinforcement learning training is highly sensitive to consistency in policy evaluation. In PPO and RLHF pipelines, when the same policy is evaluated with different batch sizes, a change in ReduceScatter reduction ordering due to sharding can inject floating‑point noise into gradient or reward signals, destabilizing policy updates.

- **verl** ([github.com/verl-project/verl](https://github.com/verl-project/verl)): A mainstream open‑source RLHF or PPO framework that provides a `full_determinism` configuration option and explicitly sets `HCCL_DETERMINISTIC=1` to guarantee reproducible collective operations.
- **DeepSpeed‑Chat** and derivative frameworks: Require deterministic reductions in RLHF training to keep reward model training consistent across identical inputs.

#### 3. Inference Consistency and Batch Invariance

In large‑model serving, users expect the same prompt to always return the same output. However, dynamic batching means a prompt can be grouped with different neighbours on each request. Without deterministic collective communication, floating‑point reduction order can vary with batch composition, breaking this invariance.

- **vLLM Batch Invariance**: The vLLM project explicitly calls out that non‑deterministic all‑reduce backends (e.g., NCCL) can cause different logits for the same prompt depending on batch mates. Their batch invariance guarantee relies on deterministic communication to ensure "the output for a given prompt is the same regardless of what other prompts are in the batch." ([Motivation](https://docs.vllm.ai/en/latest/features/batch_invariance/#motivation), [Ascend Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/batch_invariance.html))
- **SGLang**: Provides an `--enable-deterministic-inference` flag that forces deterministic computation and communication ordering, making inference outputs fully reproducible across different batch sizes and request arrival patterns. ([SGLang deterministic inference](https://sgl-project.github.io/advanced_features/deterministic_inference.html))
- **OpenAI Community**: Practitioners have long struggled with non‑deterministic GPU operations in production LLM inference, where bit‑for‑bit reproducibility is expected by end‑users and essential for debugging. ([Defeating Nondeterminism in LLM Inference](https://community.openai.com/t/defeating-nondeterminism-in-llm-inference/1358623))

#### 4. Ecosystem API and Framework Support

The demand for determinism is reflected in the official APIs and configuration flags of major ML frameworks:

- **PyTorch**: `torch.use_deterministic_algorithms(True)` requires all operations – including collectives – to produce the same output given the same input on the same hardware or software. ([PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html))
- **HuggingFace Transformers or Diffusers**: Provide a standardised `enable_full_determinism()` function that sets `NCCL_DETERMINISTIC=1`, `CUBLAS_WORKSPACE_CONFIG`, and other variables.
- **LlamaFactory**: Large‑model fine‑tuning framework offering an `enable_full_determinism(seed)` interface for reproducible distributed training.
- **ByteDance VeOmni**: Enforces `--train.enable_full_determinism true` in CI tests, making deterministic collectives a gate for code acceptance.

### Limitations of Existing Batch-Invariant Algorithms in HCCL

HCCL currently provides two batch-invariant algorithms:

| Algorithm | Use Case | Limitation |
|-----------|----------|------------|
| **Mesh + Local Reduce** | Small messages (< several MB) | Low bandwidth utilization for large messages |
| **RHD (Recursive Halving-Doubling)** | Large messages | Utilizes only approximately 50% of available bandwidth (only half the nodes communicate per round) |

On the A3 server topology (SIO + HCCS hybrid interconnect), RHD cannot simultaneously utilize SIO and HCCS links, resulting in insufficient bandwidth utilization for large message scenarios.

### Value of BIRS

The BIRS algorithm is designed for the 2D topology characteristics of A3 servers, maintaining batch invariance while:

- **First round**: Performing SendReduce over SIO links (cross X-axis reduction)
- **Subsequent rounds**: Simultaneously utilizing SIO (reduction) and HCCS (intermediate result transfer) links
- Achieving near-optimal bandwidth utilization, with only the first round not fully utilizing bandwidth

## Detailed Design

### 1. Overall Architecture

The BIRS algorithm is integrated into HCCL as an experimental feature, isolated from existing algorithms through an independent code path and build option.

```text
HCCL
├── src/ops/reduce_scatter/          # Existing ReduceScatter implementation
│   └── reduce_scatter_op.cc/.h      # Entry function (with BIRS dispatch logic added)
│
├── experimental/ops/                # Experimental features directory (new)
│   ├── op_common/                   # Common infrastructure
│   │   ├── op_common_experimental.cc/.h    # Experimental op common logic (ProcessA3, etc.)
│   │   ├── template/                # Experimental algorithm template base class
│   │   │   └── alg_template_base_experimental.cc/.h
│   │   └── topo/                    # Experimental topology utilities
│   │       └── topo_experimental.cc/.h
│   │
│   └── reduce_scatter/              # ReduceScatter experimental algorithms
│       ├── reduce_scatter_op_experimental.cc/.h  # Experimental entry (MatchBIRS dispatch)
│       └── birs/                    # BIRS algorithm implementation
│           ├── reduce_scatter_birs_executor.cc/.h # Executor layer (resource calc, scheduling)
│           ├── reduce_scatter_executor_base.cc/.h # Executor base class
│           └── template/
│               ├── reduce_scatter_birs.cc/.h      # Core algorithm template (communication loop)
│               └── reduce_scatter_birs_inter.cc/.h # Intermediate result handling
│
└── test/st/algorithm/testcase/
    └── reduce_scatter_testcase_a3.cc  # A3 platform test cases (new)
```

**Data Flow**:

```text
User calls HcclReduceScatter()
    │
    ├── HCCL_BIRS_ENABLE != TRUE → Take existing HcclReduceScatterInner() path
    │
    └── HCCL_BIRS_ENABLE == TRUE
        │
        └── ReduceScatterExperimental()
            │
            ├── Parameter validation (reuse existing CheckReduceScatterInputPara, etc.)
            │
            └── ReduceScatterOutPlaceCustom()
                │
                └── ProcessA3()
                    │
                    └── ReduceScatterBIRSExecutor::KernelRun()
                        │
                        └── ReduceScatterBIRS::RunAsync()
                            │
                            ├── Preprocess()       — Preprocessing (slice calc, channel validation)
                            ├── Main comm loop     — SIO SendReduce + HCCS transfers
                            └── FinalStep()        — Local tree reduction + output copy
```

### 2. Interface Design

#### 2.1 Environment Variables

| Environment Variable | Values | Description |
|---------------------|--------|-------------|
| `HCCL_BIRS_ENABLE` | `TRUE` / `FALSE` (default) | Enable the BIRS algorithm. When set to `TRUE`, ReduceScatter calls are routed to the BIRS implementation |

#### 2.2 Build Options

Run the following command from the root directory of the hccl repository:

```bash
# host + device + experimental
bash build.sh --pkg --full --experimental
```

```cmake
option(ENABLE_EXPERIMENTAL "Enable experimental features" OFF)
```

To enable experimental functions, use the `--experimental option`. This sets the compilation flag `-DENABLE_EXPERIMENTAL=ON`, which in turn causes the `experimental/ops/` subdirectory to be compiled. Disabled by default, with no impact on existing builds.

#### 2.3 API Compatibility

BIRS introduces no new user-facing APIs. Users call the standard `HcclReduceScatter()` interface, with algorithm selection entirely controlled by environment variables:

```c
// No user code changes required — just set the environment variable to enable
// export HCCL_BIRS_ENABLE=TRUE
HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
```

### 3. Data Structures

#### 3.1 Logical 2D Topology Layout

BIRS constructs a logical 2D layout over the A3/16P topology:

```
rankSizeX = 2                          // X-axis direction (SIO links)
rankSizeY = rankSize / rankSizeX       // Y-axis direction (HCCS links)
```

Each rank maintains the following topology information:

| Member | Type | Description |
|--------|------|-------------|
| `sio_rank` | `u32` | SIO peer rank (`rank XOR 1`) |
| `hccs_ranks` | `vector<u32>` | Peer rank list along HCCS direction |
| `hccs_neighbour_rank` | `vector<u32>` | SIO neighbor ranks of HCCS peers |
| `sio_link` | `ChannelInfo` | SIO communication channel |
| `hccs_links` | `vector<ChannelInfo>` | HCCS communication channel list |
| `hccs_links_reversed` | `vector<ChannelInfo>` | Reversed HCCS channels (for receiving) |

#### 3.2 Scratch Memory Layout

BIRS uses scratch memory to store intermediate reduction results (IM), with a strided layout to satisfy the 910B minimum slice alignment requirement:

```
localStrideSize = RoundUp(sliceSize, HCCL_MIN_SLICE_ALIGN_910B)

Scratch buffer is divided into 2 regions with N slots each: Region A is used to accumulate intermediate results for HCCS,
Region B is used for sendReduce over SIO.

Scratch Memory:
┌─────────────────────────────────────────────┐
│ IM[0]: offset = 0 * localStrideSize         │  ← Region A intermediate result
├─────────────────────────────────────────────┤
│ IM[1]: offset = 1 * localStrideSize         │  ← Region A intermediate result
├─────────────────────────────────────────────┤
│ ...                                         │
├─────────────────────────────────────────────┤
│ IM[N]: offset = N * localStrideSize         │  ← Region A intermediate result
├─────────────────────────────────────────────┤
│ SIO[0]: offset = (N+1) * localStrideSize    │  ← Region B SIO-intermediate result
├─────────────────────────────────────────────┤
│ SIO[1]: offset = (N+2) * localStrideSize    │  ← Region B SIO-intermediate result
├─────────────────────────────────────────────┤
│ ...                                         │
├─────────────────────────────────────────────┤
│ SIO[N]: offset = 2 * N * localStrideSize    │  ← Region B SIO-intermediate result
└─────────────────────────────────────────────┘
```

#### 3.3 Thread Model

BIRS uses a 3-thread parallel model:

| Thread | Role | Responsibility |
|--------|------|----------------|
| `mainThread` | Main thread | SIO SendReduce, final local reduction |
| `subThreads[0]` | HCCS sub-thread | HCCS link Send or Notify operations |
| `subThreads[1]` | Copy sub-thread | Pre-copy of next round's input data |

Inter-thread synchronization is performed via `PreSyncInterThreads` / `PostSyncInterThreads`.

### 4. Key Logic

#### 4.1 Algorithm Overview

The core property of the BIRS algorithm is **batch invariance**: the order of reduction additions on each rank is strictly identical, regardless of batch size or memory slicing.

**Notation**:
- `S(d, i)`: The i-th slice of the input message on device d
- `rankSizeX = 2`, `rankSizeY = rankSize / 2`
- `sio_rank = rank XOR 1` (SIO peer)
- `hccs_ranks[i] = (rank + rankSizeX * i) % rankSize` (HCCS peer sequence)

#### 4.2 Main Communication Loop

```
// Initial: copy the input slice corresponding to the first HCCS peer into scratch memory
LocalCopy(input[S(hccs_ranks[0])], scratch[IM_0])

for round in 0 ... hccs_ranks.size():

    // ── Sub-thread 0: HCCS transfer (when round > 0) ──
    if round > 0:
        Notify(sio → hccs_ack)
        Wait(hccs_ack)
        Send(scratch[IM_{round-1}] → hccs_peer[round-1])
        Notify(data_signal)
        Wait(data_signal)

    // ── Main thread: SIO SendReduce ──
    Notify(sio_ack)
    Wait(sio_ack)
    SendReduce(
        local:  input[S(hccs_neighbour_rank[round])],  // or S(sio_rank) in the last round
        remote: scratch[IM_round on sio_peer]
    ) → scratch[IM_round on sio_peer]
    Notify(data_signal)
    Wait(data_signal)

    // ── Sub-thread 1: Pre-copy next round's data ──
    if round < hccs_ranks.size() - 1:
        LocalCopy(input[S(hccs_ranks[round+1])], scratch[next_slot])
```

#### 4.3 Final Reduction (FinalStep)

After all rounds complete, each rank holds `rankSizeY` intermediate results in scratch memory. These are merged via a **tree-based local reduction**:

```
// Collect all intermediate result offsets
vec = [IM_0, IM_1, ..., IM_{rankSizeY-1}]  // this rank's result is at the correct position

// Tree reduction (guarantees deterministic addition order)
for stride in 1, 2, 4, ...:
    for i in stride, stride+stride, ...:
        LocalReduce(vec[i] → vec[i - stride])

// Copy final result to output
LocalCopy(vec[0] → outputMem)
```

The tree reduction guarantees deterministic addition ordering: for `rankSizeY = 4`, the reduction order is `(IM_0 + IM_1) + (IM_2 + IM_3)`, independent of rank id. ReduceScatterBIRS() supports reduction for rankSize <= 16, for larger rankSize it is recommended to use ReduceScatterBIRSInter().

### 5. Compatibility Considerations

#### 5.1 Backward Compatibility

- **Fully backward compatible**: BIRS is disabled by default (`HCCL_BIRS_ENABLE` defaults to `FALSE`), with no impact on existing ReduceScatter behavior.
- **Build isolation**: Experimental code resides in an independent `experimental/` directory, controlled by the `ENABLE_EXPERIMENTAL` build flag, and is excluded from compilation by default.
- **No API changes**: The user-facing API (`HcclReduceScatter`) remains unchanged; algorithm selection is transparent to users.

#### 5.2 Applicability Conditions

The BIRS algorithm currently has the following constraints:

| Constraint | Description |
|------------|-------------|
| Platform | A3 servers only (SIO + HCCS hybrid topology) |
| rankSize | Must be even (`rankSize % 2 == 0`), typical values: 4, 8, 16 |
| Communication domain | Both Intra-server and Inter-server are supported |
| Data alignment | Slice sizes must satisfy `HCCL_MIN_SLICE_ALIGN_910B` alignment requirements |

ReduceScatterBIRS() is the recommended choice for single-server A3 scenario (rankSize <= 16), ReduceScatterBIRSInter() is chosen automatically for multi-server A3 scenario.

When conditions are not met, the workflow exit and log error messages using hccl. The user must follow the recommendations in the error logs or manually adjust the parameters to comply with the restrictions.

#### 5.3 Rollout Strategy

1. **Phase 1** (current): As an experimental feature, with dual gating via `ENABLE_EXPERIMENTAL=ON` compile flag + `HCCL_BIRS_ENABLE=TRUE` runtime flag.
2. **Phase 2** (post-validation): Remove compile-time gating, retain only environment variable control.
3. **Phase 3** (post-stabilization): Automatically select BIRS as the default algorithm when conditions are met; users can disable via `HCCL_BIRS_ENABLE=FALSE`.

### 6. Test Plan

#### 6.1 Functional Correctness Testing

- **New test file**: `test/st/algorithm/testcase/reduce_scatter_testcase_a3.cc`
- **Test dimensions**:
  - Different rankSize values (4, 8, 16)
  - Different data types (FP16, FP32, BF16)
  - Different reduction operations (SUM, MAX, MIN, PROD)
  - Different message sizes (from KB-level to tens of MB)

#### 6.2 Batch Invariance Verification

- Execute ReduceScatter with the same input data but different batch sizes
- Verify that output results are bit-identical

#### 6.3 Performance Testing

- Compare against the RHD algorithm, measuring Task Duration across different message sizes
- Expected: For message sizes >= 16MB, BIRS achieves up to 25% improvement over RHD.
- Note: At the moment of this RFC creation kernel submission mechanism in HCCL is slower than the one of HCOMM, so 25% performance improvement applies only to operator execution time (without submission overhead).

#### 6.4 Regression Testing

- Ensure all existing ReduceScatter test cases are unaffected when `HCCL_BIRS_ENABLE=FALSE` (default)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| BIRS only available for specific rankSize (even numbers) | Odd rank scenarios cannot use BIRS | `MatchBIRS()` check auto-falls back to existing algorithms; document constraints clearly |
| Experimental code may introduce stability issues | Affects overall HCCL reliability | Dual gating (compile + runtime) isolation; independent `experimental/` directory; disabled by default |
| Additional scratch memory overhead | Increased memory usage for large messages | Requires `2 * rankSizeY × localStrideSize` scratch space; pre-allocated via `CalcResRequest` |
| A3 topology assumption (SIO + HCCS) may not apply to other platforms | Cross-platform compatibility | Algorithm explicitly bound to A3 topology characteristics; other platforms require independent adaptation |

## Alternative Approaches

N/A

## Open Questions

1. **AllReduce extension**: Batch-invariant AllReduce which follows the same ideas will be submitted in separate PR
2. **Efficient support for arbitrary rank enumeration**: Current solution assumes default rank enumeration where rankID of SIO neighbour of RankX can be calculated as (RankX XOR 1). In case of other rank enumerations BIRS is functional but doesn't deliver performance advantage over RHD. Efficient support for custom enumerations have already been implemented and will be submitted in the next PR.

---

## Review Records

The review process takes place in the PR comment section. For detailed review comments, refer to the corresponding PR:

- PR: [cann/hccl#657](https://gitcode.com/cann/hccl/pull/657)
- Issues: [cann/hcomm#139](https://gitcode.com/cann/hcomm/issues/139), [cann/hccl#96](https://gitcode.com/cann/hccl/issues/96)
