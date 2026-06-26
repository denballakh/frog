# FrogLang Project Notes

## Maintenance Instruction

Agents MUST update this `AGENTS.md` when introducing substantial project changes, including language features, compiler pipeline behavior, tests, commands, or repository workflow changes.

Agents SHOULD periodically check whether this file is incorrect, stale, or incomplete while working. If it is, they MUST update it in the same change rather than leaving follow-up documentation work for the user.

Agents MUST update user-facing docs under `docs/` when making significant user-facing changes to language behavior, CLI behavior, examples, or workflow.

Agents may add possible future improvements to `TODO.md` only after mandatory user approval. Do not add speculative TODOs without explicit approval.

If the error message or log line is incorrect, misleading, useless or in any other way nonhelpful - agent SHOULD attempt to alleviate that.

## Project Overview

FrogLang is a small stack-based, concatenative, statically typed language implemented in Python. The implementation can interpret Frog directly or translate Frog IR to C and compile it with `gcc`.

The language and implementation are inspired by Porth. Frog programs use postfix stack operations, explicit stack-effect procedure signatures, compile-time imports and macros, and block keywords such as `proc`, `macro`, `if`, `else`, `while`, `do`, `end`, and `let`.

## Repository Layout

- `frog/__init__.py`: Main implementation for tokenization, IR compilation, typechecking, interpretation, and C translation.
- `frog/__main__.py`: CLI entrypoint, REPL, subprocess helpers, and build/run orchestration.
- `frog/types.py`: Core dataclasses/enums for tokens, instructions, value classes, contracts, procedures, stacks, and pretty-printing.
- `frog/logs.py`: Logging, diagnostics, exit codes, source locations, and fatal helper functions.
- `frog/sb.py`: Persistent-ish `StringBuilder` used by pretty-printing and C code generation.
- `examples/*.frog`: Example Frog programs. Generated `examples/*.c` and `examples/*.exe` are build artifacts.
- `examples/00_empty.frog`: Empty program smoke test.
- `examples/01_simple.frog`: Basic stack arithmetic, debug, and print demo.
- `examples/02_while.frog`: While loop, nested if/else, and arithmetic demo.
- `examples/03_fib.frog`: Fibonacci sequence using procedures and stack rotation.
- `examples/04_procs.frog`: Small procedure composition and loop demo.
- `examples/05_is_prime.frog`: Prime-checking procedures and boolean logic demo.
- `examples/06_let.frog`: Local binding demo with `let`.
- `examples/07_rule30.frog`: Rule 30 ASCII cellular automaton using byte buffers, pointer arithmetic, memory reads/writes, and `putc`.
- `examples/08_gcd_grid.frog`: Euclidean GCD rendered as a 40x40 coprimality grid using `#` for coprime coordinates and space otherwise.
- `docs/README.md`: Documentation index.
- `docs/language.md`: User-facing FrogLang language reference.
- `docs/testing.md`: Test snapshot workflow and review process.
- `TODO.md`: User-approved future improvements and cleanup ideas.
- `test/__main__.py`: Snapshot test generator/runner. It runs example files, CLI cases, inline code snippets, and multi-file import-system cases.
- `test/snapshots/**/*.out`: Markdown-style snapshot output files produced by `python -m test`. Snapshots embed tested source or CLI arguments with captured output.
- `test/tmp_fs/`: Temporary filesystem tree created by tests for inline code and multi-file cases; generated `.c`/`.exe` files under it are build artifacts.
- `ide/vscode/`: Minimal VS Code language grammar for `.frog` files.
- `devenv.nix`, `devenv.yaml`, `.envrc`: Nix/devenv environment setup.
- `justfile`: Project command recipes.

## Environment

- Python requirement is `>=3.13`.
- The devenv shell provides optimized Python 3.13 plus `mypy`, `basedpyright`, `black`, `types-tqdm`, `tqdm`, `git`, and Nix language support.
- Building Frog files requires `gcc` because `python -m frog build` emits C and invokes `gcc`.
- The README uses `py -m frog`, but local commands in this repo generally use `python -m frog`.

## Common Commands

- Show available recipes: `just`
- Typecheck with mypy and basedpyright: `just typecheck`
- Format Python with Black: `just fmt`
- Run typecheck and format: `just check`
- Run the full test suite, including typecheck/format first: `just test`
- Show regenerated snapshot diffs: `just show-diff`
- Approve regenerated snapshot diffs after careful review: `just approve-diff`
- Start REPL: `just repl`
- Run Frog CLI through just: `just cli <args>`
- Remove generated C/exe artifacts: `just clean`

Useful direct commands:

- CLI help: `just cli -h` or `python -m frog --help`
- Interpret a file: `python -m frog run examples/01_simple.frog`
- Interpret inline code: `python -m frog run -c '1 2 + print'`
- Build a file: `python -m frog build examples/01_simple.frog`
- Build and run: `python -m frog build -r examples/01_simple.frog`
- Trace logging: `python -m frog -l TRACE run examples/01_simple.frog`

## Formatting And Typechecking

- Black config is in `pyproject.toml`: line length `120`, target `py313`, and `skip-string-normalization = true`.
- Keep the existing single-quote style in Python; Black is configured not to normalize strings.
- Typechecking uses both `mypy .` and `basedpyright .`.
- Pyright mode is `recommended`, with `reportAny`, `reportExplicitAny`, and `reportCallInDefaultInitializer` disabled.
- The code intentionally uses modern Python typing features such as `type Stack = ...`, `typing.Self`, `typing.override`, dataclass pattern matching, and exhaustive `assert_never` checks.

## Testing Nuances

- `just test` is the expected and recommended full verification command
- Do not run `just check` and `python -m test` separately as a substitute for `just test`; the test suite uses shared generated files and separate/parallel runs can race.
- `just test` regenerates `test/snapshots/**/*.out` by capturing stdout from many scenarios, then fails if the snapshot directory differs from git, including untracked files.
- Snapshots are self-contained Markdown-style `.out` files. They embed the Frog source or CLI command under test before the captured output.
- Inline code and multi-file import cases run both `frog run` and `frog build -r`. If the captured output body is identical after the top-level command line, the snapshot lists both commands and stores the shared output once; otherwise it keeps separate run and build outputs.
- Import-system behavior tests live in `test/__main__.py` as multi-file cases. They write temporary directory trees under `test/tmp_fs/` and cover imported procedures, macros, reexports, root-relative paths, conflicts, cycles, and rejected syntax.
- Use `just show-diff` to inspect snapshot changes.
- Use `just approve-diff` to approve snapshot changes ONLY IF YOU ARE ABSOLUTELY SURE the regenerated outputs are correct.
- After behavior changes, inspect the regenerated snapshot `.out` files to confirm the new output is intentional.
- The test runner also builds and runs examples through the C backend, so `gcc` must be available.
- `test/tmp_fs/` is created during tests and removed at the end; failed runs can leave generated artifacts there.
- Use `just clean` after builds/tests if generated `.c`/`.exe` files are not intended to remain.

## CLI Behavior

- Entrypoint is `python -m frog`.
- Global options are parsed before the subcommand. `-l` accepts `ERROR`, `WARN`, `INFO`, or `TRACE`.
- Subcommands are `run`, `build`, and `repl`.
- `run` accepts `-c CODE` or a file path, then tokenizes, compiles, typechecks, and interprets.
- `build FILE` writes generated C beside the source using `.c`, then compiles to `.exe` by default or `-o FILE` if provided.
- `build -r FILE` also executes the compiled binary.
- Exit codes are defined in `frog/logs.py`: `0` ok, `1` normal error, `2` usage error, `3` internal error.
- `frog.__main__.run_frog` is a test helper that calls `main()` directly and intentionally skips the first three argv-like strings.

Current CLI help output:

```text
$ just cli -h
Usage: py -m frog [OPTIONS] SUBCOMMAND <ARGS>

Options:
  -h --help                   print this help message
  -l <level>                  log level: ERROR,WARN,INFO,TRACE
Subcommands:
  run [OPTIONS]             interpret
    -c CODE                   code to interpret
       FILE                   file to interpret
  build [OPTIONS] FILE      build
    FILE                      file to build
    OPTIONS:
      -o FILE                 where to put built binary
      -r                      also run the binary
  repl                      start a Read-Eval-Print-Loop
```

## Compiler Pipeline

- `tokenize(text, filename)` creates `Token` objects with `Loc` source positions and emits TRACE logs when enabled.
- `compile(toks)` builds a root-relative import graph, strips import and macro declarations, resolves per-module symbol scopes, then lowers modules into an `IR` containing explicit `Module` objects and `Proc` instruction lists.
- The supported import syntax is `from "path.frog" import name`, `from "path.frog" import name as alias`, and grouped whitespace-separated imports such as `from "path.frog" import ( x y z )`. Wildcards, commas, and `import "path.frog" as mod` are rejected for now.
- Import paths are resolved relative to the root file being compiled, not relative to the importing module. Use explicit paths such as `"pkg/math.frog"` for subdirectory files.
- Imported files contribute procedures and macros. Imported top-level instructions are ignored and only the root module's `main` runs.
- Imported names are reexported, so facade modules can import a symbol and expose it to their importers.
- Macro declarations are collected with whole-module scope before the remaining code is compiled. Macro expansion is module-aware: imported and reexported macros resolve helper words in the module where the macro was defined. Recursive macro expansion is rejected.
- Top-level instructions create an implicit `main` proc; an empty program still gets an empty `main`.
- Explicit `proc main -- do ... end` must have no inputs and no outputs.
- `typecheck(ir)` simulates stack effects and rejects stack underflows, unknown words, wrong contracts, bad branch/loop stack states, and non-empty final stacks.
- `interpret(ir)` executes the IR directly and prints debug/runtime output.
- `translate(ir)` emits C code using `StringBuilder`, then the CLI compiles it with `gcc`.
- Generated C sanitizes Frog procedure names into valid C identifiers for `proc_*`, `ret_*`, and related result variable prefixes, so punctuation in procedure names does not directly leak into C symbols.

## Language Semantics

- User-facing language documentation lives in `docs/language.md`; update it when changing Frog syntax, semantics, intrinsics, examples, diagnostics that users see, or backend-visible behavior.
- `macro name <body> end` records `<body>` as a compile-time token sequence in the defining module. Macro bodies may use function-body block constructs such as `if`, `while`, and `let`, but not nested `proc`, nested `macro`, or import declarations.
- `let a b c do ... end` binds visible stack values in source order: after `1 2 3`, `let a b c do` binds `a = 1`, `b = 2`, and `c = 3`. The implementation emits reverse-order pops to achieve this.

## Implementation Conventions And Gotchas

- `frog/__init__.py` is large and intentionally keeps the pipeline together; prefer minimal targeted edits unless a refactor is explicitly needed.
- When adding an intrinsic, update all relevant places together: `IntrinsicType`, `INTRINSIC_TO_INTRINSIC_TYPE`, the `expect_enum_size(IntrinsicType, ...)` check, typechecker behavior, interpreter behavior, C translator behavior, tests, docs, and optionally VS Code grammar. Keep concrete language behavior documented in user-facing docs rather than adding one-off feature facts here.
- When adding a value class, update `ValueClsType`, the `expect_enum_size(ValueClsType, ...)` checks, typechecking, interpretation, C type mapping, stack-copy logic, printing, casts, and tests.
- When adding a keyword, update `KeywordType`, `KW_TO_KWT`, parser/compiler handling, macro body validation if the keyword affects block syntax, tests, docs, and `ide/vscode/frog_grammar.json`.
- Error paths often call `error(...)`, which prints diagnostics then `sys.exit(...)`; tests rely on captured stdout and exit-code lines from helpers.
- Internal consistency failures should use `unreachable`, `typecheck_has_a_bug`, or `notimplemented` from `frog/logs.py` as appropriate.
- Do not treat generated `.c` or `.exe` files as authoritative source. They are build/test artifacts even if some currently exist in the tree.
- `StringBuilder.__str__` mutates/collapses its internal linked chunks; copying is available through `copy()` or `[::]`.
- Generated C uses simple structs named `ret_<proc>` and functions named `proc_<proc>`, with imported module procedure names sanitized into module-qualified C identifiers and generated variable names globally uniqued per translation.

## VS Code Grammar

- The grammar is a small TextMate JSON package for `.frog` files.
- If language keywords, types, operators, word-like intrinsics, comments, or literals change, update `ide/vscode/frog_grammar.json` as part of the same change.
- The existing repository key is spelled `punctiation`; preserve or fix carefully because references currently use that spelling.

## Working Tree Hygiene

- The repository ignores generated `*.c`, `*.exe`, Python caches, mypy cache, `.devenv*`, `.direnv`, and local env files.
- Before finalizing code changes, prefer `just precommit` when feasible. For docs-only changes, a lighter verification may be enough.
- If tests regenerate files under `test/snapshots/`, review those diffs carefully because they are the effective behavioral snapshots.
