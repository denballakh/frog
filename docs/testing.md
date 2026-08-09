# Testing

The snapshot suite is orchestrated by `test/__main__.py` through:

```sh
just test
```

`just test` runs formatting, Python typechecking, native bootstrap checks, and snapshot generation. It fails if `test/snapshots/` has tracked or untracked changes afterward.

`compiler/frogc.frog` is the sole Frog compiler, typechecker, and C emitter. `compiler/frogc_cli.c` orchestrates native processes and files, while `test/__main__.py` materializes cases, invokes the CLI in subprocesses, and renders snapshots. Python is test-only; there is no Python language implementation or interpreter.

`just bootstrap-check` verifies that `compiler/frogc.c` is a fixed-point bootstrap seed and compiles and runs focused native compiler fixtures under `test/bootstrap/` with strict C11 warnings.

Snapshots are Markdown-style `.out` files. They embed the Frog source or CLI command being tested, followed by captured output, so a snapshot diff can usually be reviewed without opening the fixture source separately.

Snapshot groups:

- `test/snapshots/examples/`: one snapshot per `examples/*.frog` file.
- `test/snapshots/cli/`: grouped CLI argument behavior.
- `test/snapshots/code/`: grouped inline Frog snippets.
- `test/snapshots/imports/`: one snapshot per multi-file import-system case.

Each example, inline snippet, and multi-file case runs once through the native `build/frogc run` path. That path compiles Frog to C, compiles the C program, and executes the resulting binary.

Inline cases use immutable `SourceSpec` values. `body` is mechanically indented inside an explicit `proc main -- do ... end`; `before_main` and `after_main` hold declarations whose placement matters; `raw_source` is reserved for malformed or top-level structural probes. Snapshots embed the fully materialized source, not the concise fields. Multi-file roots use the same representation, while imported module files remain verbatim source.

One focused CLI snapshot exercises `build -r` under `test/tmp_fs/`. Additional assertions force GCC to fail after Frog has directly regenerated the C output, verify that a subsequent successful build is deterministic, and verify lexical import resolution through a symlinked root source. `run` reuses ignored scratch artifacts under `build/`, so it does not publish artifacts beside examples.

`test/tmp_fs/` is recreated for a run and removed in a `finally` block. Each CLI subprocess has a bounded timeout; exceeding it terminates the process group and fails the test run rather than producing an approvable snapshot.

Useful commands:

```sh
just show-diff
just approve-diff
```

Only run `just approve-diff` after carefully reviewing the regenerated snapshots.
