# Testing

Run the complete suite with:

```sh
just test
```

This checks Python formatting and types, validates the VS Code grammar,
checks the compiler bootstrap fixed point and repository-wide Frog formatting,
runs the Frog-owned test corpus, and checks the remaining host-specific CLI
policies.

## Repository format check

Run the tracked-source formatting invariant with:

```sh
just format-check
```

The check enumerates tracked `*.frog` files with Git's NUL-delimited output.
Every valid ordinary module must already be byte-identical to `frogc fmt`
output. Entries in `test/formatter_exclusions.tsv` must remain tracked, unique,
and sorted. Intentional compiler-error fixtures must still fail formatting with
empty standard output, so stale exclusions and newly unclassified failures are
rejected.

## Frog test corpus

`test/runner.frog` runs the language, compiler, standard-library, example,
import, and CLI cases. Successful programs are compiled to C, built with strict
C11 warnings, and executed. Tests compare exact output, diagnostics, and exit
status, and also check selected generated-C properties.

CLI cases also verify that `frogc check [FILE]` analyzes every function without
emitting C, using standard input when `FILE` is omitted. Formatter cases cover
line-boundary stack indentation, folded physical lines, structured regions,
comments, multiline literals, LF/CRLF/lone-CR preservation, idempotence,
file-relative imports, invalid input, preserved macro declarations, and the
no-in-place-write contract. They also cover polymorphic macro call sites and
expanded and folded `elif` arms. The tests pin the current version of `frogc
inspect [FILE]` output, including standard input,
relative imports, resolved cross-module targets, and typed control-flow blocks,
and require semantic failures to leave standard output empty. Format-5 cases
also verify lossless token, comment, and whitespace coverage, explicit syntax
regions, escaped lexeme text, and combined `--builtins --lexemes` inspection.
Cursor cases verify command-line errors, byte and trivia selection, end of file,
canonical declaration and import-alias identities, overload selection, local
and builtin shadowing, macro/type/local resolution precedence, compound
owner/member spans, specialized macro contexts, and empty standard output after
analysis failure.

The case manifests are:

- `test/regression_cases.frog` for fixture-based language, compiler, generated-C,
  and standard-library regressions;
- `test/inline_cases.frog` for concise source snippets;
- `test/import_cases.frog` for multi-file import behavior;
- `test/cli_cases.frog` for ordinary command-line behavior;
- `test/example_cases.frog` for every checked-in example.

Fixtures live under `test/cases/`. Inline helpers wrap ordinary snippet bodies
in `func main -- do ... end`; malformed whole-program cases provide their source
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
