# HCCL Tuner Plugin Example

## Overview

This directory provides a reference implementation of the HCCL Tuner Plugin, demonstrating how to modify the cost table via an external `.so` plugin that reads a JSON configuration, thereby influencing the Selector's algorithm choice.

## Build

```bash
source /usr/local/Ascend/cann/set_env.sh
make
```

Artifact: `hccl_tuner_example.so`

The first build automatically downloads nlohmann/json (header-only) to `third_party/`. For offline environments, manually place `third_party/nlohmann/json.hpp` beforehand; `make` will skip the download if detected.

## Usage

```bash
export HCCL_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
export HCCL_TUNER_CONFIG_FILE=/path/to/hccl_tuner_config.json
```

The plugin searches for the configuration file in the following order, loading the first readable one:

1. `$HCCL_TUNER_CONFIG_FILE` (skipped if not set)
2. `./hccl_tuner_config.json`
3. `/etc/hccl/hccl_tuner_config.json`

## JSON Configuration Format

```json
{
  "version": 1,
  "op_types": {
    "allreduce": {
      "rules": [
        {
          "match": {
            "min_ranks": 8, "max_ranks": 8,
            "min_bytes": 0, "max_bytes": 65536,
            "data_type": "fp16"
          },
          "engine": "aicpu",
          "executor": "sole",
          "template": "nhr",
          "cost": 0.0
        }
      ]
    }
  }
}
```

### Top-level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | int | Yes | Configuration format version, currently must be `1` |
| `op_types` | object | Yes | Rule set organized by operator type |

### Match Conditions (all AND, first-match-wins)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `min_ranks` / `max_ranks` | uint32 | Yes | Communication domain rank count range |
| `min_bytes` / `max_bytes` | size_t | Yes | Data size range (bytes) |
| `data_type` | string | No | Data type (`int8`/`int16`/`int32`/`int64`/`uint8`/`uint16`/`uint32`/`uint64`/`fp16`/`float16`/`fp32`/`float32`/`fp64`/`float64`/`bfp16`/`bfloat16`) |
| `comm_name` | string | No | Communication domain name (substring match) |
| `min_npus_per_server` / `max_npus_per_server` | uint32 | No | NPUs per server range |
| `min_servers` / `max_servers` | uint32 | No | Server count range |
| `min_pods` / `max_pods` | uint32 | No | Pod count range |
| `min_super_pods` / `max_super_pods` | uint32 | No | Super pod count range |
| `buffer_size` | uint64 | No | Communication domain buffer size (exact match) |

### Match Behavior

- `engine` / `executor` / `template` (required) specify the target position in the cost table
- `template` for single-level algorithms is a single template name (e.g., `mesh`); for multi-level algorithms it is the lowercased concatenation of the remaining string after the executor in algName (e.g., `meshconcurnhrnhr`)
- `cost` (required) sets the cost value at that position; `0.0` means optimal
- The Selector chooses the minimum cost from the cost table to determine the algorithm

### engine Available Values

`aicpu` / `ccums` / `ccusched` / `aiv` / `dpu`

### executor Available Values

`sole` / `sequence` / `parallel` / `pipeline` / `concur`

### template Available Values

Single-level algorithms (enum-validated):

`mesh` / `mesh2die` / `meshoneshot` / `meshtwoshot` / `meshconcur` / `meshmultilink` / `meshchunk` / `meshchunktwoshot` / `nhr` / `nhrmultilink`

Multi-level algorithms (concatenation string, not enum-validated; typos caught at runtime via warning):

Take the lowercased remaining string after the executor in algName, e.g., `AicpuAllReduceSequenceMeshConcurNHRNHR` → `meshconcurnhrnhr`.

```json
{ "engine": "aicpu", "executor": "sequence", "template": "meshconcurnhrnhr", "cost": 0.0 }
```

### Supported op_type

`allreduce` / `allgather` / `broadcast` / `reduce` / `reduce_scatter` / `scatter` / `alltoall` / `alltoallv`

> The available values for engine / executor / template / op_type above are consistent with the dimension definitions of the HCCL algorithm selector.

## Test

```bash
cd test
make
./test_plugin
```

## Plugin Interface

The plugin must export the symbol `hcclTunerPlugin_v1` (type `hcclTunerFuncs_v1_t`), containing two function pointers: `init` and `getCollInfo`. The HCCL core loads the plugin via `dlsym` to obtain this symbol.

See `include/hccl_tuner_plugin.h` for details.

## Runtime Examples

The following are typical outputs from `./test_plugin`, demonstrating the plugin's behaviors.

### Rule Hit and Cost Override

AllReduce with 8 ranks, 4096B, fp32 matches a rule. The target algorithm's cost is changed from 100.0 to 0.0 (optimal preference). The Selector will pick this algorithm:

```
[TunerDFX] rule hit: opType=2 nBytes=4096 dataType=3 ruleIdx=0/2
  engine=aicpu executor=sole template=meshoneshot cost=0.000000
[TunerDFX] modify: algName=AicpuAllReduceSoleMeshOneShot
  cost 100.000000 -> 0.000000
```

Multi-level algorithms are also supported. The `template` field is the lowercased remainder string after the executor in the algorithm name:

```
[TunerDFX] rule hit: opType=2 nBytes=4096 dataType=3 ruleIdx=0/1
  engine=aicpu executor=sequence template=meshconcurnhrnhr cost=0.000000
[TunerDFX] modify: algName=AicpuAllReduceSequenceMeshConcurNHRNHR
  cost 5.000000 -> 0.000000
```

### No Match — No Intervention

4 ranks is outside the required 8–16 range. No rule matches, the cost table retains CostModel values:

```
[TunerDFX] no rule matched: opType=2 nBytes=4096 dataType=3
```

### Target Algorithm Disabled (cost<0) — Skip

The target entry has cost=-1 (already disabled). The plugin does not modify it. In `SelectMinCost`, cost<0 is treated as filtered and excluded from selection:

```
[TunerDFX] rule hit: opType=2 nBytes=4096 dataType=3 ruleIdx=0/2
  engine=aicpu executor=sole template=meshoneshot cost=0.000000
[TunerDFX] skip disabled: algName=AicpuAllReduceSoleMeshOneShot cost=-1.000000
[TunerDFX] rule matched but no entry modified
```

### Schema Validation Failed — No Intervention

When the config contains typos, missing required fields, invalid enums, or any other schema errors, the plugin does not intervene at all (safe degradation):

```
Schema: unknown field 'mtach' in rule, skipping
Schema: rule missing required field 'match'
Schema: invalid engine 'invalid_engine'
Schema: min_ranks(16) > max_ranks(8)
tuner config loaded, schemaErrors=4
Schema validation failed (4 errors), plugin will not intervene
```
