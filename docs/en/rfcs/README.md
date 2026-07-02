# RFC Document Directory

This directory contains the technical solution design documents (RFC - Request for Comments) for the HCCL repository. These documents are used to align solutions and document design decisions before coding implementation.

## Directory Structure

- `0000-template.md` - RFC writing template
- `INDEX.md` - RFC number registry, listing all allocated RFC numbers
- `NNNN-xxx-xxx.md` - RFC documents (4-digit number plus short description)

## Naming Rules

RFC file naming format: `{4-digit number}-{short description}.md`

For example: `0001-add-new-feature.md`

- Number: 4-digit zero-padded (0001 to 9999)
- Description: English lowercase, hyphen-separated, concise

## Numbering Mechanism (Core)

1. **Number-claiming PR** (lightweight): Modify only [INDEX.md](./INDEX.md) by adding one row to reserve a number.
2. **RFC document PR** (heavyweight): Write the RFC document and submit it for review. The number is already locked through the number-claiming PR.

**Number claiming rules**:

- Sequential assignment, **smallest unused number** first.
- Never reused. Merged numbers are not recycled even if the RFC is later superseded.
- For details, see [INDEX.md](./INDEX.md).

## RFC Lifecycle

1. **Requirement phase**: Submit a Requirement-type Issue and wait for the SIG group to accept it.
2. **Number-claiming PR**: Add a placeholder row in [INDEX.md](./INDEX.md) with the status set to `reserved`. Submit the number-claiming PR.
3. **Number-claiming PR merged**: The number is reserved, and you can start writing the RFC.
4. **Writing phase**: Write the system solution following the [RFC template](./0000-template.md).
5. **Review phase**: Submit the RFC document PR for review. Modify the solution based on feedback.
6. **Decision phase**:
   - **Merged**: The Maintainer passes the review, adds `/lgtm` and `/approve`, and merges. Update the status in INDEX.md to `accepted`.
   - **Closed**: The review is not passed. The PR is closed. The number remains `reserved` and is not recycled. The author can restart the review process.
7. **Implementation phase**: The merged RFC serves as the implementation contract. Code PRs must follow the RFC solution.

## Supersession

When an RFC implementation is superseded by a subsequent RFC:

- Add the following annotation at the end of the superseded RFC document: `> Superseded by 00NN`
- Update the corresponding row status in [INDEX.md](./INDEX.md) from `accepted` to `superseded`.
- Do not modify the original number.

## Related Links

- [Contribution Guide](../../../CONTRIBUTING_en.md)
- [RFC Template](./0000-template.md)
- [RFC Number Registry](./INDEX.md)
- [SIG Meeting](https://etherpad-cann.meeting.osinfra.cn/p/sig-hccl)
