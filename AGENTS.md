# FrogLang Project Notes

## Maintenance Instruction

Agents MUST update this `AGENTS.md` when introducing substantial project changes, including language features, compiler pipeline behavior, tests, commands, or repository workflow changes.

Agents SHOULD periodically check whether this file is incorrect, stale, or incomplete while working. If it is, they MUST update it in the same change rather than leaving follow-up documentation work for the user.

Agents MUST update user-facing docs under `docs/` when making significant user-facing changes to language behavior, CLI behavior, examples, or workflow.

Agents may add possible future improvements to `TODO.md` only after mandatory user approval. Do not add speculative TODOs without explicit approval.

If the error message or log line is incorrect, misleading, useless or in any other way nonhelpful - agent SHOULD attempt to alleviate that.

## Project Overview

FrogLang is a small stack-based, concatenative, statically typed language compiled to C. `compiler/frogc.frog` implements the compiler, typechecker, C emitter, and CLI process/file orchestration. Python is test-only; the repository has no Python language implementation or Frog interpreter.

The language and implementation are inspired by Porth. Frog programs use postfix stack operations, explicit stack-effect procedure signatures, nominal records, compile-time imports and macros, and block keywords such as `proc`, `record`, `macro`, `if`, `else`, `while`, `do`, `end`, and `let`.

## Repository Layout

- `compiler/frogc.frog`: The Frog-written compiler, typechecker, deterministic C emitter, and CLI. Filter mode reads the root source from stdin; file commands preserve lexical source-relative import resolution.
- `compiler/frogc.c`: Checked-in fixed-point C bootstrap seed generated from `compiler/frogc.frog`; this is an authoritative bootstrap artifact, not a disposable build output.
- `examples/*.frog`: Example Frog programs. Generated `examples/*.c` and `examples/*.exe` are build artifacts.
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
- `test/__main__.py`: Snapshot orchestration for example files, CLI cases, inline snippets, and multi-file imports, plus black-box build-artifact assertions. It invokes the Frog-written CLI in subprocesses and contains no language implementation.
- `test/bootstrap/`: Focused native compiler fixtures and shell harnesses run by `just bootstrap-check`.
- `test/snapshots/**/*.out`: Markdown-style snapshot output files produced by `python -m test`. Snapshots embed tested source or CLI arguments with captured output.
- `test/tmp_fs/`: Temporary filesystem tree created by tests for inline code and multi-file cases; generated `.c`/`.exe` files under it are build artifacts.
- `ide/vscode/`: Minimal VS Code language grammar for `.frog` files.
- `devenv.nix`, `devenv.yaml`, `.envrc`: Nix/devenv environment setup.
- `justfile`: Project command recipes.

## Environment

- Python requirement is `>=3.13`.
- The devenv shell provides optimized Python 3.13 plus `mypy`, `basedpyright`, `black`, `git`, and Nix language support.
- Running or building Frog files requires `gcc`: the Frog-written CLI compiles Frog to C and then compiles the C program.

## Common Commands

- Show available recipes: `just`
- Typecheck with mypy and basedpyright: `just typecheck`
- Format Python with Black: `just fmt`
- Run typecheck and format: `just check`
- Run the full test suite, including typecheck/format first: `just test`
- Build the checked bootstrap seed: `just frogc-seed`
- Verify the checked seed, source, and next two generations are byte-identical: `just bootstrap-check`
- Regenerate the checked seed after a verified compiler-source change: `just bootstrap-update`
- Show regenerated snapshot diffs: `just show-diff`
- Approve regenerated snapshot diffs after careful review: `just approve-diff`
- Run Frog CLI through just: `just cli <args>`
- Remove generated root/example/test C/exe artifacts: `just clean`

Useful direct commands:

- CLI help: `build/frogc -h`
- Compile and run a file: `build/frogc run examples/01_simple.frog`
- Compile and run inline source: `build/frogc run -c 'proc main -- do 1 2 + print end'`
- Build a file: `build/frogc build examples/01_simple.frog`
- Build and run: `build/frogc build -r examples/01_simple.frog`
- Use the compiler as a stdin-to-stdout filter: `build/frogc < source.frog > source.c`

## Formatting And Typechecking

- Black config is in `pyproject.toml`: line length `120`, target `py313`, and `skip-string-normalization = true`.
- Keep the existing single-quote style in Python; Black is configured not to normalize strings.
- Typechecking uses both `mypy .` and `basedpyright .`.
- Pyright mode is `recommended`, with `reportAny`, `reportExplicitAny`, and `reportCallInDefaultInitializer` disabled.
- The Python test runner intentionally uses modern typing features such as `type SourceInput = ...`, frozen dataclasses, pattern matching, and exhaustive `assert_never` checks.

## Testing Nuances

- `just test` is the expected and recommended full verification command
- Do not run `just check` and `python -m test` separately as a substitute for `just test`; the test suite uses shared generated files and separate/parallel runs can race.
- `just test` regenerates `test/snapshots/**/*.out` by capturing stdout from many scenarios, then fails if the snapshot directory differs from git, including untracked files.
- Snapshots are self-contained Markdown-style `.out` files. They embed the Frog source or CLI command under test before the captured output.
- Each example, inline, and multi-file corpus case runs once through the `build/frogc run` path.
- Inline cases use immutable `SourceSpec` values to materialize an explicit `proc main -- do ... end`; declaration-order and malformed-structure cases use the appropriate structural fields or verbatim raw source.
- Import-system behavior tests live in `test/__main__.py` as multi-file cases. They write temporary directory trees under `test/tmp_fs/` and cover imported procedures, macros, reexports, root-relative paths, conflicts, cycles, and rejected syntax.
- Use `just show-diff` to inspect snapshot changes.
- Use `just approve-diff` to approve snapshot changes ONLY IF YOU ARE ABSOLUTELY SURE the regenerated outputs are correct.
- After behavior changes, inspect the regenerated snapshot `.out` files to confirm the new output is intentional.
- One focused CLI case exercises `build -r`. Additional Python assertions verify direct-output behavior across a forced GCC failure, deterministic regeneration, successful replacement, and lexical imports through a symlinked root source.
- CLI `build` test artifacts live under `test/tmp_fs/`, which is recreated for a run and removed in a `finally` block. CLI `run` reuses ignored `build/frog-run.c` and `build/frog-run.exe` scratch artifacts.
- `just bootstrap-check` compiles its focused fixtures with strict C11 warnings and compares their output, in addition to checking compiler fixed-point equality.
- Use `just clean` after builds/tests if generated `.c`/`.exe` files are not intended to remain.

## CLI Behavior

- Entrypoint is `build/frogc` (or `just cli <args>`).
- With no arguments, it is a compiler filter: it reads Frog source from standard input and writes generated C to standard output.
- Subcommands are `run` and `build`; each has `-h`/`--help`.
- `run` accepts `-c CODE` or one file path. It invokes the compiler core in a child process, writes reusable C/executable scratch artifacts under `build/`, and executes the binary.
- `build FILE` compiles Frog directly to a source-adjacent `.c`, then compiles C directly to an `.exe`; `-o FILE` selects a different executable destination.
- `build -r FILE` runs the resulting executable.
- CLI argument parsing, path construction, build policy, process setup, and exit-status forwarding are implemented in Frog in `compiler/frogc.frog`. Generated-C runtime adapters expose only the POSIX ABI details that the scalar C FFI cannot represent directly.

Current CLI help output:

```text
$ build/frogc -h
Usage:
  frogc < source.frog > source.c
  frogc <command> [options]

Commands:
  run [-c CODE | FILE]       compile and run Frog source
  build [-o FILE] [-r] FILE  compile Frog source to a binary
```

## Compiler Pipeline

- `compiler/frogc.frog` is the sole lexer, parser/declaration scanner, module loader, typechecker, macro expander, and C emitter.
- The compiler reads root source bytes from stdin and writes generated C to stdout. Imported files are loaded relative to the root compiler process's working directory.
- The supported import syntax is `from "path.frog" import name`, `from "path.frog" import name as alias`, and grouped whitespace-separated imports such as `from "path.frog" import ( x y z )`. Wildcards, commas, and `import "path.frog" as mod` are rejected for now.
- Import paths are resolved relative to the root file being compiled, not relative to the importing module. Use explicit paths such as `"pkg/math.frog"` for subdirectory files.
- Imported files contribute procedures and macros. Imported top-level instructions are ignored and only the root module's `main` runs.
- Imported names are reexported, so facade modules can import a symbol and expose it to their importers.
- Macro declarations are collected with whole-module scope before the remaining code is compiled. Macro expansion is module-aware: imported and reexported macros resolve helper words in the module where the macro was defined. Recursive macro expansion is rejected.
- Every root program must define exactly one `proc main -- do ... end` with no inputs or outputs. Empty sources, declaration-only sources without `main`, and root top-level executable instructions are invalid.
- Typechecking occurs while procedures and expanded macros are compiled to C; failures include stack underflow, unknown words, contract mismatches, invalid control-flow stack shapes, and non-empty final stacks.
- Generated C uses a runtime cell stack and numeric procedure symbols, so source punctuation does not become a C identifier.
- `compiler/frogc.c` must remain a checked fixed point: compiling `compiler/frogc.frog` with the seed and recompiling it with the result must reproduce the same C bytes.
- `bootstrap-update` compiles candidate compiler generations as standalone binaries and invokes their no-argument stdin-to-stdout filter mode.

## Language Semantics

- User-facing language documentation lives in `docs/language.md`; update it when changing Frog syntax, semantics, intrinsics, examples, diagnostics that users see, or generated-C behavior.
- `macro name <body> end` records `<body>` as a compile-time token sequence in the defining module. Macro bodies may use function-body block constructs such as `if`, `while`, and `let`, but not nested `proc`, nested `macro`, or import declarations.
- `let a b c do ... end` binds visible stack values in source order: after `1 2 3`, `let a b c do` binds `a = 1`, `b = 2`, and `c = 3`. The implementation emits reverse-order pops to achieve this.
- `elif` is lowered to nested existing IF/ELSE/END instructions; one source `end` closes the whole chain, and the no-`else` path participates in stack-shape checking.
- `read-file` consumes a UTF-8 path as `ptr int` and produces file bytes, byte length, and a success boolean as `ptr int bool`. On failure it returns zero length and `false`; the returned data pointer must not be dereferenced.
- `args` has stack effect `-- ptr int` and exposes the generated program's raw C `argv` followed by `argc`, including `argv[0]`; `@ptr` loads and `!ptr` stores one pointer-sized entry as `ptr`.
- `record Name field Type ... end` defines a nominal pointer-backed record. `Name:alloc` allocates uninitialized storage, `Name:sizeof` exposes its Cell-based byte size, and `Name.field`/`Name.field!` provide statically typed access.
- Record fields occupy one eight-byte Cell in declaration order. Record-valued fields store handles, and only explicit `ptr`/record casts cross the nominal boundary.

## Implementation Conventions And Gotchas

- Keep language semantics and CLI policy in `compiler/frogc.frog`; generated-C runtime adapters should remain narrow ABI primitives rather than command parsers or build-policy implementations.
- When adding an intrinsic, update native recognition, type-stack behavior, emitted C/runtime support, bootstrap and snapshot coverage, user-facing docs, and optionally the VS Code grammar.
- String literals lower to a UTF-8 byte pointer and byte length (`ptr int`); generated globals and macro expansion must retain the defining module's literal identity.
- Record type IDs are program-global and nominal. Imported aliases and reexports must retain the defining record identity; type-level operations use `:` and field operations use `.`.
- Frog `int` is an `int64_t` cell in generated C. Fixed-width memory accesses must remain byte-safe through `memcpy` helpers.
- When adding a keyword, update native declaration/body scanning, macro validation, compilation, tests, docs, and `ide/vscode/frog_grammar.json`.
- User-facing compiler failures use stable `frogc: ...` diagnostics on standard error. Keep exact diagnostics covered by focused fixtures when practical.
- `test/__main__.py` uses `SourceSpec` to materialize concise inline bodies, declarations before or after `main`, and verbatim malformed structural probes. It must write and snapshot the materialized source.
- Do not treat generated `.c` or `.exe` files as authoritative source, except for the intentional bootstrap seed `compiler/frogc.c`. Other generated files remain disposable build/test artifacts.
- CLI `build` intentionally writes outputs directly and provides no locking or rollback transaction.

## VS Code Grammar

- The grammar is a small TextMate JSON package for `.frog` files.
- If language keywords, types, operators, word-like intrinsics, comments, or literals change, update `ide/vscode/frog_grammar.json` as part of the same change.
- The existing repository key is spelled `punctiation`; preserve or fix carefully because references currently use that spelling.

## Working Tree Hygiene

- The repository ignores generated `*.c`, `*.exe`, Python caches, mypy cache, `.devenv*`, `.direnv`, and local env files.
- `compiler/frogc.c` is the explicit exception to the generated-C ignore rule. Update the generated seed only with `just bootstrap-update`, whose fixed-point comparison must pass first.
- Before finalizing code changes, run `just test` when feasible. For docs-only changes, a lighter verification may be enough.
- If tests regenerate files under `test/snapshots/`, review those diffs carefully because they are the effective behavioral snapshots.
