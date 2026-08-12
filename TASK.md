# Compiler simplification

- [x] Replace manual record-array growth copies with `memory-copy`.
- [x] Represent normalized path components with a record instead of parallel arrays.
- [x] Simplify `token-type-prefix-equals`, dot-component checks, `type-snapshot`, and recursive C metadata emitters.
- [x] Deduplicate the standard-library byte-copy implementation used by subprocess capture.
- [x] Reconcile documented test policy and remove duplicated or mutating check recipes.
- [x] Regenerate the checked-in C compiler.
- [x] Pass `just bootstrap-check` and `just test`.
- [x] Commit and push the completed change on `master`.

Preserve the pre-existing uncommitted change in `stdlib/json.frog`.
