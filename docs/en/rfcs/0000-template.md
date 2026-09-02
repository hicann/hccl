# RFC Template

- Start Date: (Fill in the date in YYYY-MM-DD format)
- RFC PR Number: (Associated PR number)
- Related Issue: (Associated Requirement Issue number)

---

## Summary

What is being done + why it is done this way. Reviewers should grasp the core decision and its rationale within 30 seconds of reading this section; everything that follows is supporting material.

## Background and Motivation

Expand on the problem and scenarios.

- Why is this feature or modification needed?
- What problem does it solve?
- What are the expected use cases?

## Terminology

Align the key terms involved in this RFC to avoid ambiguity during review. Mandatory for cross-module or cross-repo RFCs; may be omitted for single-module changes.

| Term | Meaning |
|------|---------|

## Architecture and Interface Contract

This section answers "how the system is organized + what the external contract is"; it does not cover module-internal implementation.

### Overall Architecture

Architecture diagram of the modules involved in this RFC, responsibilities of each module, and runtime sequence diagrams.

### External Interfaces

Describe the external boundary interfaces from the user's perspective, including function call interfaces and configuration; the expected behavior and effect of each interface.

### Dependency Interfaces

Which interfaces of hccl / hcomm / other CANN components this depends on.

## Impact Analysis

- Performance impact (latency, throughput, memory/resource usage)
- Scope of effect on existing functionality (which modules/interfaces are affected)
- Impact on build, dependencies, and release

## Compatibility Considerations

- Does it affect backward compatibility?
- Is a feature switch required?
- What is the phased rollout strategy?
- Compatibility impact of external and dependency interface changes

## Detailed Design

This section covers only module-internal implementation. Organize by module; for cross-layer features, split into separate module subsections per layer, do not mix them.

Each module subsection includes:
- Module responsibility (one sentence)
- Core data structures (UML may be used)
- Key logic and algorithms
- Internal interfaces with other modules

## Algorithm Design

Fill in this section if a core algorithm is involved; otherwise it may be omitted.

- Algorithm principle and mathematical model
- Complexity analysis (time / space / communication volume)
- Correctness argument

## Test Plan

UT / ST / on-board test plans, verifying functional correctness and compatibility.

## Risk Assessment

- Potential risk points
- Risk mitigation measures

## Alternative Solutions

Describe other solutions considered and their advantages and disadvantages.

## Open Issues

Issues that have not been resolved during the design phase or require further discussion.

