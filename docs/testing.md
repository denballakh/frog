# Testing

The test suite is driven by `python -m test`, and the usual entrypoint is:

```sh
just test
```

`just test` runs formatting and typechecking, regenerates snapshots under `test/snapshots/`, and fails if the snapshot directory has any tracked or untracked changes afterward.

Snapshots are Markdown-style `.out` files. They embed the Frog source or CLI command being tested, followed by captured output, so a snapshot diff can usually be reviewed without opening the fixture source separately.

Snapshot groups:

- `test/snapshots/examples/`: one snapshot per `examples/*.frog` file.
- `test/snapshots/cli/`: grouped CLI argument behavior.
- `test/snapshots/code/`: grouped inline Frog snippets.
- `test/snapshots/imports/`: one snapshot per multi-file import-system case.

Inline and multi-file cases run both `frog run` and `frog build -r`. When both commands produce the same output body, the snapshot lists both commands and stores the shared output once. When interpreter and C backend behavior differ, both outputs are kept.

Useful commands:

```sh
just show-diff
just approve-diff
```

Only run `just approve-diff` after carefully reviewing the regenerated snapshots.
