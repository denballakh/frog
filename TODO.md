# TODO

This file tracks user-approved future improvements.
Agents may add items here only after explicit user approval.
When an item from this list is fixed/implemented - remove it from this list.

## New language features

- support including other source files

## Diagnostics And Debugging

- Add a debug mode that prints named stack effects around each word and intrinsic.
- Improve contract mismatch errors to show the expected stack suffix and actual stack suffix in source order.
- Normalize user-facing intrinsic names in errors, for example `+` instead of `IntrinsicType.ADD`.
- Reduce noisy internal source locations in user-facing diagnostics, especially for `notimplemented` paths whose `[LOC] __init__.py:...` line changes when implementation code moves.
