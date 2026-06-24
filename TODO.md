# TODO

This file tracks user-approved future improvements. Agents may add items here only after explicit user approval.

## Diagnostics And Debugging

- Add a debug mode that prints named stack effects around each word and intrinsic.
- Improve contract mismatch errors to show the expected stack suffix and actual stack suffix in source order.
- Validate procedure names during Frog compilation so backend-invalid names fail before C code generation.
- Improve documentation and tests for stack binding and stack-effect order when examples expose confusing cases.
- Normalize user-facing intrinsic names in errors, for example `+` instead of `IntrinsicType.ADD`.
- Reduce noisy internal source locations in user-facing diagnostics, especially for `notimplemented` paths whose `[LOC] __init__.py:...` line changes when implementation code moves.
