# TODO

This file tracks user-approved future improvements.
Agents may add items here only after explicit user approval.
When an item from this list is fixed/implemented - remove it from this list.

## New language features

- Add multiline string literals so large templates, including generated C fragments, do not need to be encoded as long single-line literals.
- Add compile-time constants that evaluate expressions during compilation and expand into one or more literals instead of reevaluating the expression at every runtime use.

## Diagnostics And Debugging

- Add a debug mode that prints named stack effects around each word and intrinsic.
- Improve contract mismatch errors to show the expected stack suffix and actual stack suffix in source order.
- Use source spelling for intrinsic names in diagnostics, for example `+` rather than an internal compiler label.
