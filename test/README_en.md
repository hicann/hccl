# HCCL Test

This directory contains the HCCL test code, which is divided into system tests (ST) and unit tests (UT).

## Directory Structure

```
test/
├── st/                         # System Test
│   ├── algorithm/              # Algorithm analyzer tests
│   │   ├── testcase/           # Algorithm test cases
│   │   ├── utils/              # Test utility code
│   │   │   └── src/
│   │   │       ├── aicpu/          # AICPU-related stubs
│   │   │       ├── common/         # Common utilities
│   │   │       │   ├── exception/  # Exception handling
│   │   │       │   └── utils/      # Utility functions
│   │   │       ├── hccl_depends_stub/  # HCCL dependency interface stubs
│   │   │       ├── hccl_proxy/     # Simulated communicator implementation
│   │   │       │   ├── communicator/
│   │   │       │   └── topo_model/
│   │   │       ├── hccl_verifier/  # Verifier
│   │   │       │   ├── mem_conflict_check/   # Memory conflict checking
│   │   │       │   ├── semantics_check/     # Semantics checking
│   │   │       │   ├── singletask_check/   # Single-task checking
│   │   │       │   └── task_graph_generator/# Task graph generation
│   │   │       ├── sim_world/     # Simulation world implementation
│   │   │       └── ut/            # Algorithm analyzer UT tests
│   │   ├── figures/            # Test illustration images
│   │   ├── CMakeLists.txt
│   │   ├── README.md           # Algorithm analyzer detailed guide
│   │   └── build.sh            # Compilation and execution script
└── ut/                         # Unit Test
    └── common/
        └── prepare_ut_env/     # UT environment preparation code
```

## Test Types

### System Test (ST)

System tests mainly verify the correctness of HCCL collective communication algorithm logic, including memory operation validation and semantics validation.

#### Algorithm Analyzer

The algorithm analyzer verifies algorithm logic and memory operation functions by simulating the HCCL single-operator execution flow.

**Principle:**

1. The algorithm analyzer stubs the dependencies (hcomm and runtime interfaces) of the HCCL single-operator flow to obtain the Task sequences of all ranks during algorithm execution.
2. The Task information of all ranks is formed into a **directed acyclic graph**.
3. Validation is performed based on **graph algorithms**, including memory read-write conflict checking and semantics checking.

**Core Functions:**

- **Memory conflict checking**: Analyzes whether there are possible read-write conflicts based on the synchronization situation in the Task graph.
- **Semantics checking**: Simulates Task graph execution, records data movement information, and verifies whether the data movement in the output memory meets the operator requirements.

For details, see the [Algorithm Analyzer Guide](./st/algorithm/README_en.md).

### Unit Test (UT)

Run the following commands in the repository root directory:

```bash
# Compile and run all unit test cases
bash build.sh --ut

# Compile and run all system test cases
bash build.sh --st
```
