# RFC Number Registry

This document registers all assigned RFC numbers. Before adding a new RFC, claim the **smallest unused number** from the table below.

## Number Claiming Process (Independent Reservation, Decoupled from RFC Document Creation)

1. Check the Allocated Numbers table to find the smallest unused N (usually the last row number + 1).
2. Add a new row for N in this table with the status set to `reserved`. The title and author can be placeholders.
3. Submit a **number-claiming PR** (containing only one line update in this INDEX.md).
4. After the number-claiming PR is merged, number N is reserved (status remains `reserved`). You can then start writing the RFC document.
5. The RFC document `NNNN-xxx-xxx.md` is submitted through a subsequent independent PR.
6. After the RFC document PR is merged, the status changes to `accepted`, and the RFC officially takes effect.

## Status Description

| Status | Description |
|------|------|
| `reserved` | Number is reserved (number-claiming PR merged). RFC document pending submission or review. |
| `accepted` | RFC document PR merged. The RFC is officially in effect. |
| `superseded` | Superseded by a subsequent RFC. See the `Superseded by` annotation at the end of the original document. |

## Allocated Numbers

| Number | Title | Author | Status | PR |
|------|------|------|------|-----|
| 0001 | BIRS (Batchsize Invariant ReduceScatter) for A3 | Davydov_Danil | accepted | [#1440](https://gitcode.com/cann/hccl/merge_requests/1440) |

## Numbering Rules

- **Format**: 4-digit zero-padded (0001 to 9999)
- **Never reused**: Merged numbers are not recycled even if the RFC is later superseded.
- **Sequential assignment**: In principle, numbers are not skipped. The next number is the largest used number + 1.
- **Supersession**: Add `> Superseded by 00NN` at the end of the original RFC document and update the status column in this table to `superseded`.
- **Conflict resolution**: If two people claim the same number simultaneously, the latter must rebase and change to the new smallest number.

## Related

- [RFC Template](./0000-template.md)
- [RFC Process Description](./README.md)
