# Testing

Run the complete suite with:

```sh
just test
```

This checks Python formatting and types, validates the VS Code grammar,
checks the compiler bootstrap fixed point, runs the Frog-owned test corpus, and
checks the remaining host-specific CLI policies.

## Frog test corpus

`test/runner.frog` runs the language, compiler, standard-library, example,
import, and CLI cases. Successful programs are compiled to C, built with strict
C11 warnings, and executed. Tests compare exact output, diagnostics, and exit
status, and also check selected generated-C properties.

CLI cases also verify that `frogc check [FILE]` analyzes every procedure without
emitting C, using standard input when `FILE` is omitted.

The case manifests are:

- `test/regression_cases.frog` for fixture-based language, compiler, generated-C,
  and standard-library regressions;
- `test/inline_cases.frog` for concise source snippets;
- `test/import_cases.frog` for multi-file import behavior;
- `test/cli_cases.frog` for ordinary command-line behavior;
- `test/example_cases.frog` for every checked-in example.

Fixtures live under `test/cases/`. Inline helpers wrap ordinary snippet bodies
in `proc main -- do ... end`; malformed whole-program cases provide their source
verbatim.

Run only this corpus with:

```sh
just frog-regressions
```

## Host policy checks

`test/__main__.py` is limited to CLI policies that require host environment or
filesystem control. It verifies behavior across a forced GCC failure, checks
lexical import resolution through a symlinked root source, and verifies that a
compiler invoked through a slashless `argv[0]` can show help but cannot locate
the standard library. Each launched process group, including the Frog runner and
its children, has a 120-second timeout and process-group termination. Frog
source used by these checks lives under `test/cases/host_policy/`.

Python is test-only. The compiler, CLI, and all language/compiler case
declarations and assertions are implemented in Frog; there is no Python Frog
compiler or interpreter.

Run only the host policy checks after building `build/frogc` with:

```sh
just frogc-seed
python -m test
```

`test/tmp_fs/` is recreated for the host checks and removed afterward. Run
`just clean` to remove ignored build and test artifacts under `build/`,
`examples/`, and `test/`. Tracked fixtures and `compiler/frogc.c` are kept.

## Bootstrap check

`just bootstrap-check` verifies that the checked-in `compiler/frogc.c`, stage 2,
and stage 3 are byte-identical. Language behavior belongs in the Frog test
corpus rather than in a separate bootstrap suite.
