# FrogLang Project Notes

## Maintenance Instruction

If you notice that AGENTS.md file is wrong - update it. And if you change the project substantially - update the AGENTS.md automatically.

## Project Overview

FrogLang is a small stack-based, concatenative, statically typed language implemented in Python. The implementation can interpret Frog directly or translate Frog IR to C and compile it with `gcc`.

The language and implementation are inspired by Porth. Frog programs use postfix stack operations, explicit stack-effect procedure signatures, and block keywords such as `proc`, `if`, `else`, `while`, `do`, `end`, and `let`.

## Repository Layout

- `frog/__init__.py`: Main implementation for tokenization, IR compilation, typechecking, interpretation, and C translation.
- `frog/__main__.py`: CLI entrypoint, REPL, subprocess helpers, and build/run orchestration.
- `frog/types.py`: Core dataclasses/enums for tokens, instructions, value classes, contracts, procedures, stacks, and pretty-printing.
- `frog/logs.py`: Logging, diagnostics, exit codes, source locations, and fatal helper functions.
- `frog/sb.py`: Persistent-ish `StringBuilder` used by pretty-printing and C code generation.
- `examples/*.frog`: Example Frog programs. Generated `examples/*.c` and `examples/*.exe` are build artifacts.
- `test/__main__.py`: Golden-output test generator/runner. It runs example files, CLI cases, and inline code snippets.
- `test/*.out`: Golden output files produced by `python -m test`.
- `test/tmp.c` and `test/tmp.exe`: Generated test build artifacts.
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
- Run tests, including typecheck/format first: `just test`
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

- `just test` deletes `test/*.out` first, then runs `python -m test`.
- `python -m test` does not compare against committed goldens itself; it regenerates `test/*.out` by capturing stdout from many scenarios.
- After behavior changes, inspect the regenerated `.out` files to confirm the new output is intentional.
- The test runner also builds and runs examples through the C backend, so `gcc` must be available.
- `test/tmp.frog` is created during tests and unlinked at the end; failed runs can leave generated artifacts.
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
  run [OPTIONS]             interpre
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
- `compile(toks)` lowers tokens into an `IR` containing `Proc` objects and `Instruction` lists.
- Top-level instructions create an implicit `main` proc; an empty program still gets an empty `main`.
- Explicit `proc main -- do ... end` must have no inputs and no outputs.
- `typecheck(ir)` simulates stack effects and rejects stack underflows, unknown words, wrong contracts, bad branch/loop stack states, and non-empty final stacks.
- `interpret(ir)` executes the IR directly and prints debug/runtime output.
- `translate(ir)` emits C code using `StringBuilder`, then the CLI compiles it with `gcc`.

## Language Semantics

- Supported runtime value classes are `int`, `bool`, `ptr`, and `type`, though parser-level procedure signatures currently only accept `int` and `bool`.
- Integer literals are non-negative decimal chunks. Negative values are produced by operations, not by signed literal syntax.
- `true` and `false` are bool literals.
- Character literals push integer codepoints. Only exactly one raw character is accepted; backslash escape handling is not implemented despite tests covering escaped-looking inputs.
- String literals tokenize, but compilation currently reports `not implemented: string literals`.
- `//` starts a line comment only when tokenized as its own whitespace-delimited chunk.
- Intrinsics include arithmetic, bitwise, logic, comparisons, stack manipulation, `cast`, `print`, and `?` debug.
- `?` logs the stack at compile time during typechecking and at runtime during interpretation; it is omitted in C codegen.
- `print` prints `[PRINT] ...` in the interpreter, but generated C prints raw values without the `[PRINT]` prefix.
- `if <cond> do <then> [else <else>] end` requires the condition to leave exactly one bool and both branches to leave the same stack shape.
- `while <cond> do <body> end` requires the condition to leave exactly one bool and the loop body to preserve the original stack shape.
- `let a b c do ... end` pops values into bindings in reverse word order at compile time emission, then `LOAD_BIND` reuses the bound `StackEntry`; `end` emits matching unbinds.
- Procedure calls are statically checked against declared stack contracts. Return values are modeled as struct fields in generated C.
- Casts currently allow same-type, `int`/`bool`, `bool`/`int`, `int`/`ptr`, and `ptr`/`int` conversions.

## Implementation Conventions And Gotchas

- `frog/__init__.py` is large and intentionally keeps the pipeline together; prefer minimal targeted edits unless a refactor is explicitly needed.
- When adding an intrinsic, update all relevant places together: `IntrinsicType`, `INTRINSIC_TO_INTRINSIC_TYPE`, the `expect_enum_size(IntrinsicType, ...)` check, typechecker behavior, interpreter behavior, C translator behavior, tests, and optionally VS Code grammar.
- When adding a value class, update `ValueClsType`, the `expect_enum_size(ValueClsType, ...)` checks, typechecking, interpretation, C type mapping, stack-copy logic, printing, casts, and tests.
- When adding a keyword, update `KeywordType`, `KW_TO_KWT`, parser/compiler handling, tests, and `ide/vscode/frog_grammar.json`.
- Error paths often call `error(...)`, which prints diagnostics then `sys.exit(...)`; tests rely on captured stdout and exit-code lines from helpers.
- Internal consistency failures should use `unreachable`, `typecheck_has_a_bug`, or `notimplemented` from `frog/logs.py` as appropriate.
- Do not treat generated `.c` or `.exe` files as authoritative source. They are build/test artifacts even if some currently exist in the tree.
- `StringBuilder.__str__` mutates/collapses its internal linked chunks; copying is available through `copy()` or `[::]`.
- Generated C uses simple structs named `ret_<proc>` and functions named `proc_<proc>`, with generated variable names globally uniqued per translation.

## VS Code Grammar

- The grammar is a small TextMate JSON package for `.frog` files.
- If language keywords, types, operators, comments, or literals change, update `ide/vscode/frog_grammar.json` as part of the same change.
- The existing repository key is spelled `punctiation`; preserve or fix carefully because references currently use that spelling.

## Working Tree Hygiene

- The repository ignores generated `*.c`, `*.exe`, Python caches, mypy cache, `.devenv*`, `.direnv`, and local env files.
- Before finalizing code changes, prefer `just precommit` when feasible. For docs-only changes, a lighter verification may be enough.
- If tests regenerate `test/*.out`, review those diffs carefully because they are the effective golden outputs.
