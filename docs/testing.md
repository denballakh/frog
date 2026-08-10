# Testing

The snapshot suite is orchestrated by `test/__main__.py` through:

```sh
just test
```

`just test` runs formatting, Python typechecking, compiler fixed-point checks, native regression cases, and snapshot generation. It fails if `test/snapshots/` has tracked or untracked changes afterward.

`compiler/frogc.frog` implements the compiler, typechecker, C emitter, and process/file CLI orchestration. `test/__main__.py` invokes the Python regression runner, materializes snapshot cases, invokes the Frog-written CLI in subprocesses, and renders snapshots. Python is test-only; there is no Python language implementation or interpreter.

`just bootstrap-check` verifies that `compiler/frogc.c`, stage 2, and stage 3 are byte-identical. Language and runtime regressions are normal test cases rather than a separate bootstrap suite.

`test/regressions.py` runs the fixtures under `test/cases/`. It uses the compiler's stdin-to-stdout mode, compiles successful output with strict C11 warnings, links fixture-local helper C where required, and checks exact output, exit status, diagnostics, and selected generated-C properties.

Snapshots are Markdown-style `.out` files. They embed the Frog source or CLI command being tested, followed by captured output, so a snapshot diff can usually be reviewed without opening the fixture source separately.

Snapshot groups:

- `test/snapshots/examples/`: one snapshot per `examples/*.frog` file.
- `test/snapshots/cli/`: grouped CLI argument behavior.
- `test/snapshots/code/`: grouped inline Frog snippets.
- `test/snapshots/imports/`: one snapshot per multi-file import-system case.

Each example, inline snippet, and multi-file case runs once through the native `build/frogc run` path. That path compiles Frog to C, compiles the C program, and executes the resulting binary.

Top-level `examples/*.frog` files are discovered automatically. Adding an example therefore requires reviewing and committing its generated snapshot, but no test-runner registration.

Inline cases use immutable `SourceSpec` values. `body` is mechanically indented inside an explicit `proc main -- do ... end`; `before_main` and `after_main` hold declarations whose placement matters; `raw_source` is reserved for malformed or top-level structural probes. Snapshots embed the fully materialized source, not the concise fields. Multi-file roots use the same representation, while imported module files remain verbatim source.

One focused CLI snapshot exercises `build -r` under `test/tmp_fs/`. Additional assertions force GCC to fail after Frog has directly regenerated the C output, verify that a subsequent successful build is deterministic, and verify lexical import resolution through a symlinked root source. `run` reuses ignored scratch artifacts under `build/`, so it does not publish artifacts beside examples.

`test/tmp_fs/` is recreated for a run and removed in a `finally` block. Regression artifacts, generated source, and executables remain inside that temporary tree. Each subprocess has a bounded timeout; exceeding it fails the test run rather than producing an approvable snapshot.

Useful commands:

```sh
just show-diff
just approve-diff
```

Only run `just approve-diff` after carefully reviewing the regenerated snapshots.
