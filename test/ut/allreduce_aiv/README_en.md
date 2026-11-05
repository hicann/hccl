# AllReduce AIV write-semantics core allocation tests

CPU stubs execute the production TwoShot write kernel and record copies, reductions, and flag dependencies. An interval memory model checks complete output coverage, non-overlapping plain writes, initialized reads, and the original binary reduction expression for every element. The scheduler checks that every wait can complete and every flag has a single publisher.

Coverage includes producer byte thresholds, element widths, empty and uneven slices, one reducer per original slice, direct completion publication, single/dual producer transitions, non-power-of-two ranks, the few-core fallback, and extra idle cores. Each rank uses `groupSize` reduction cores, one per original slice; other consumers perform AG copies. Tests require no intermediate completion flags, exactly one localReduce publisher per slice, and input-readiness waits only for nonempty slices. Producer selection retains its 95 KiB send and 4 KiB reduction-budget thresholds to isolate the reduction-policy comparison.

After preparing the repository UT environment, run:

```bash
bash build.sh --ut
```

The CMake target is `allreduce_aiv_core_test`. These tests validate scheduling and per-element reduction order. Floating-point operations, NPU pipeline synchronization, and performance require hardware ST validation.
