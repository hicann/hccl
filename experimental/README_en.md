# experimental/ — Developer Experiment and Contribution Directory

## 1. Purpose

The `experimental/` directory provides an **experimental space** for the HCCL community. The following table describes the differences from the main `src/` directory:

| | src/ | experimental/ |
|---|---|---|
| Goal | Production-grade code | Rapid prototype validation |
| Review | RFC + SIG review | RFC + SIG review |
| Quality | Production-grade | Prototype-grade |
| Stability | API stability guaranteed | Not guaranteed |

---

## 2. Directory Structure

```
experimental/
│
├── ops/                             # Extended HCCL communication operators
└── eco_system/                       # Ecosystem tools and peripheral innovations around HCCL
```

> Expansion rule: New categories must be added as subdirectories under the corresponding major category. To add a top-level directory, SIG discussion is required.

---

## 3. Contribution Process

### 3.1 Quick Contribution (experimental)

```
Step 1: Determine the category
    ├── Modifying the HCCL library internals? → ops/
    └── Building tools around HCCL? → eco_system/

Step 2: Create a subdirectory
    ops/<category>/<project_name>/
    or
    eco_system/<category>/<project_name>/

Step 3: Write a README.md (required)
    Include at least: motivation, design, usage, status, and limitations

Step 4: Submit a PR
    Title: [experimental] ops|eco_system/<category>/<project>: <brief description>
    Target branch: master (directly merged into experimental/)
    Review criteria:
      ✅ Correct directory location
      ✅ Complete README.md
      ✅ Code does not modify any files outside experimental/
```

---

## 4. Subdirectory Templates

### Minimum Template

```
experimental/<ops|eco_system>/<category>/<project_name>/
├── README.md          # Required
└── ...                # Other files can be organized freely
```

### Recommended Template (C++ Project)

```
experimental/<ops|eco_system>/<category>/<project_name>/
├── README.md          # Required: motivation, design, usage, status, limitations
├── CMakeLists.txt     # Recommended: independent compilation
├── src/               # Source code
├── include/           # Header files (if any)
├── test/              # Tests (recommended)
└── example/           # Usage examples (recommended)
```

---

## 5. Runtime Switch

To prevent experimental contributions from being accidentally enabled and affecting the main code, the contributed feature must be controlled through a runtime switch:

**Switch naming**: Environment variable `HCCL_EXPERIMENTAL_<NAME>=true`, where `<NAME>` uses uppercase letters separated by underscores (consistent with the feature name).

**Enablement pattern**: Provide an `IsXxxEnabled()` function at the project entry point (the function name is not mandatory), with the following internal logic:

```cpp
bool IsXxxEnabled() {
    constexpr bool xxxEnabled = false;  // Default false
    if (!xxxEnabled) return false;
    const char* env = getenv("HCCL_EXPERIMENTAL_<NAME>");
    return env && std::string(env) == "true";
}
```

- The constant value defaults to `false`. The constant has higher priority: when the constant is `false`, the function directly returns `false`.
- When the constant is `true`, check whether the environment variable is `"true"`. If so, enable the feature.
- The feature entry point must be guarded with `if (IsXxxEnabled()) { ... }`.

## 6. Maintenance Policy

| Policy | Description |
|------|------|
| **Stability** | Experimental code does not guarantee API stability and can be changed at any time. |
| **Quarterly review** | Maintainers scan quarterly and mark projects with no activity for 6 months as `stale`. |
| **Code standards** | It is recommended to follow the CANN community standards, but they are not mandatory. |
| **Issue tracking** | Experimental project issues are declared in the project README and do not occupy the main Issue tracker. |

### Stale Criteria

- No commits in the last 6 months.
- No Issue or PR activity in the last 6 months.
- The "Status" in the README indicates completion or no follow-up plans.

---
