# HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-01T03:50:50.475Z pushedAt=2026-09-01T07:26:38.141Z -->

## Function

This environment variable configures the data split ratio for the AllReduce, AllGather, ReduceScatter, Broadcast, and Reduce operators under a specific two-dimension parallel communication algorithm.

Two-dimension parallelism refers to the use case where intra-server communication (using the Mesh algorithm) and inter-server communication (using the NHR algorithm) run in parallel. In this case, HCCL splits the data to be communicated in each round into two slices, which take two parallel communication paths: one slice performs Mesh communication first and then NHR communication, and the other slice performs NHR communication first and then Mesh communication. This environment variable adjusts the sizes of the two data slices so that the longer communication path is assigned less data and the shorter path is assigned more data, thereby improving load balancing between the two paths.

Set this environment variable to a digit. Value range: \[0, 1\]. Default value: 0.5.

Note the following:

- This environment variable only acts on the allocation ratio of the two data slices for the AllReduce, AllGather, ReduceScatter, Broadcast, and Reduce operators when HCCL has already selected the two-dimension parallel communication algorithm. It is not used to select the communication algorithm.
- For the AllReduce, ReduceScatter, Broadcast, and Reduce operators, the target split ratios of data slices 0 and 1 are *the value of this environment variable* and *1 - the value of this environment variable*. These ratios are opposite for an AllGather operator.
- During actual splitting, rounding or alignment is performed based on the data type size, alignment requirements, and tail block data volume. Therefore, the actual split ratios may deviate slightly from the configured values.
- If not configured, or configured beyond 0 to 1, the system uses the default value `0.5`. If a non-numeric value is configured, the system returns an error when initializing the environment variable.
- In general, keep the default value. Change it only when you confirm that the current task is in cross-chassis two-dimension communication and you need to tune performance for a specific network topology and data volume.

Use `R` to represent the value of HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO.

![HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO](./figures/HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO.png)

When ReduceScatter and AllGather use the same `R`, the data slice corresponding to `R` differs. The reasons are as follows:

- ReduceScatter and AllGather have opposite communication semantics. ReduceScatter reduces the complete input data and scatters it to each rank, while AllGather collects the sliced data on each rank and restores it to the complete output data.
- The two-dimension parallel communication algorithm uses data slice 0 and data slice 1 for two communication paths, but the mapping between slices and communication paths is reversed for the two operators. In ReduceScatter, `R` corresponds to data slice 0, that is, the "Mesh -> NHR" communication path. In AllGather, `R` corresponds to data slice 1, that is, the "NHR -> Mesh" communication path.
- Therefore, if `R=0.6`, in ReduceScatter, the "Mesh -> NHR" communication path is allocated about 60% of the data, while in AllGather, the "NHR -> Mesh" communication path is allocated about 60% of the data.

In ReduceScatter, the size of data slice 0 is `R`, and the size of data slice 1 is `1-R`:

```text
Each round of data to be communicated  
|---------------- Data slice 0: R ----------------|------ Data slice 1: 1-R ------|  

Parallel stage 1:  
Data slice 0: Mesh  =============================>  
Data slice 1: NHR   =============================>  

Parallel stage 2:  
Data slice 0: NHR   =============================>  
Data slice 1: Mesh  =============================>  

Communication path for data slice 0: Mesh -> NHR, size R  
Communication path for data slice 1: NHR -> Mesh, size 1-R
```

If the "Mesh -> NHR" communication path is slower in the ReduceScatter operator, decrease `R`. If the "NHR -> Mesh" communication path is slower, increase `R`.

In AllGather, the size of data slice 0 is `1-R`, and the size of data slice 1 is `R`:

```text
Each round of data to be communicated  
|------ Data slice 0: 1-R ------|---------------- Data slice 1: R ----------------|  

Parallel stage 1:  
Data slice 0: Mesh  =============================>  
Data slice 1: NHR   =============================>  

Parallel stage 2:  
Data slice 0: NHR   =============================>  
Data slice 1: Mesh  =============================>  

Communication path for data slice 0: Mesh -> NHR, size 1-R  
Communication path for data slice 1: NHR -> Mesh, size R
```

If the "Mesh -> NHR" communication path is slower in the AllGather operator, increase `R`. If the "NHR -> Mesh" communication path is slower, decrease `R`.

## Configuration Example

```bash
export HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO=0.5
```

## Constraints

- This environment variable takes effect only when the AllReduce, AllGather, ReduceScatter, Broadcast, and Reduce operators select the two-dimension parallel communication algorithm. If another communication algorithm is selected based on the current topology, data volume, data type, reduce type, or operator expansion mode, this environment variable does not take effect.
- Verify the performance against the actual networking and communication data volume before adjusting this environment variable. A split ratio that is too small or too large may cause unbalanced loads on the two data slices and degrade communication performance.

## Applicable Products

Ascend 950PR/Ascend 950DT
