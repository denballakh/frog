# FrogLang Project Notes

## Maintenance Instruction

Agents MUST update this `AGENTS.md` when introducing substantial project changes, including language features, compiler pipeline behavior, tests, commands, or repository workflow changes.

Agents SHOULD periodically check whether this file is incorrect, stale, or incomplete while working. If it is, they MUST update it in the same change rather than leaving follow-up documentation work for the user.

Agents MUST update user-facing docs under `docs/` when making significant user-facing changes to language behavior, CLI behavior, examples, or workflow.

Agents may add possible future improvements to `TODO.md` only after mandatory user approval. Do not add speculative TODOs without explicit approval.

If the error message or log line is incorrect, misleading, useless or in any other way nonhelpful - agent SHOULD attempt to alleviate that.

## Project Overview

FrogLang is a small stack-based, concatenative, statically typed language compiled to C. `compiler/frogc.frog` implements the compiler, typechecker, C emitter, and CLI process/file orchestration. Python is test-only; the repository has no Python language implementation or Frog interpreter.

The language and implementation are inspired by Porth. Frog programs use postfix stack operations, explicit stack-effect procedure signatures, nominal records, tagged unions, first-class function references, compile-time imports and macros, and block keywords such as `proc`, `record`, `union`, `fn`, `macro`, `if`, `else`, `while`, `do`, `end`, and `let`.

## Repository Layout

- `compiler/frogc.frog`: The Frog-written compiler, typechecker, deterministic C emitter, and CLI. It imports the libc words it uses from `stdlib/libc.frog`. Filter mode reads the root source from stdin; file commands preserve lexical source-relative import resolution.
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
- `examples/09_records.frog`: Record allocation, field access, and a typed procedure.
- `examples/10_tagged_unions.frog`: A tagged result with checked testing and projection.
- `examples/11_c_ffi.frog`: Calls C standard-library functions through explicit C interop declarations.
- `docs/README.md`: Documentation index.
- `docs/language.md`: User-facing FrogLang language reference.
- `docs/stdlib.md`: User-facing standard-library module reference.
- `docs/testing.md`: Test suite layout and commands.
- `TODO.md`: User-approved future improvements and cleanup ideas.
- `stdlib/`: Dependency-free Frog modules. They declare external dependencies
  through explicit C interop and implement library policy in Frog.
- `stdlib/string.frog`: Literal-string comparison, byte-range helpers, and the
  manually managed growable `ByteBuffer` record.
- `stdlib/subprocess.frog`: Direct child execution with literal argv/input,
  optional child cwd, captured stdout/stderr, and explicit result ownership.
- `stdlib/test.frog`: Explicit-suite checks for booleans, integers, byte ranges,
  and strings, with failure aggregation and status-based completion.
- `test/runner.frog` and `test/framework.frog`: Frog-owned test entrypoint and assertions. The manifests in `test/*_cases.frog` cover regressions, inline snippets, imports, CLI behavior, and examples. They compile successful output with strict C11 warnings, link fixture-local helper C where required, run executables, check exact diagnostics/output/status, and enforce selected generated-C properties.
- `test/__main__.py`: Minimal Python host-policy runner. It supplies bounded process-group supervision, a forced-GCC-failure environment, and a symlinked root path; it contains no Frog language cases or implementation.
- `test/cases/`: Language, compiler, standard-library, import, and host-policy fixtures used by the test runners.
- `test/tmp_fs/`: Temporary filesystem tree created and removed by the Python host-policy checks.
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
- The Python host-policy runner uses modern Python typing and standard-library subprocess APIs.

## Testing Nuances

- `just test` is the expected and recommended full verification command
- Do not run `just check` and `python -m test` separately as a substitute for `just test`; the test suite uses shared generated files and separate/parallel runs can race.
- `just test` runs the Frog-owned corpus and then the small Python host-policy runner. Do not run its components concurrently because they share generated build files.
- Successful Frog cases compile through stdin-to-stdout mode with strict C11 warnings before execution. Cases check exact stdout, stderr, and status; compiler failures ignore partial C on stdout but require the exact diagnostic and status 1.
- The Frog runner checks that every registered case helper executes. Update `expected-test-cases` only when intentionally adding or removing a case.
- `test/inline_cases.frog` materializes concise bodies inside an explicit `proc main -- do ... end`; declaration-order and malformed structural cases use the corresponding whole-source helper.
- Multi-file import cases are checked-in fixtures under `test/cases/import_cases/`; their manifest is `test/import_cases.frog`.
- `test/__main__.py` owns only bounded process-group supervision, generated CLI artifact cleanup, forced-GCC-failure build policy, and lexical symlink-root policy. `just frog-regressions` invokes it with `--frog-only`, so the Frog runner and all nested children are terminated together on timeout. Keep Frog source for host-policy checks in `test/cases/host_policy/` rather than embedding it in Python.
- Python host-policy artifacts live under `test/tmp_fs/`, which is recreated for a run and removed in a `finally` block. CLI `run` reuses ignored `build/frog-run.c` and `build/frog-run.exe` scratch artifacts.
- `just bootstrap-check` checks only compiler fixed-point equality. The Frog regression runner compiles focused fixtures with strict C11 warnings and checks their output as part of `just frog-regressions` / `just test`.
- Use `just clean` after builds/tests if generated `.c`/`.exe` files are not intended to remain.

## CLI Behavior

- Entrypoint is `build/frogc` (or `just cli <args>`).
- With no arguments, it is a compiler filter: it reads Frog source from standard input and writes generated C to standard output.
- Subcommands are `run` and `build`; each has `-h`/`--help`.
- `run` accepts `-c CODE` or one file path. It invokes the compiler core in a child process, writes reusable C/executable scratch artifacts under `build/`, and executes the binary.
- `build FILE` compiles Frog directly to a source-adjacent `.c`, then compiles C directly to an `.exe`; `-o FILE` selects a different executable destination.
- `build -r FILE` runs the resulting executable.
- CLI argument parsing, path construction, build policy, process setup, and exit-status forwarding are implemented in Frog in `compiler/frogc.frog` over the bindings in `stdlib/libc.frog`.

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
- The compiler reads root source bytes from stdin and writes generated C to stdout. Root imports are loaded relative to the compiler process's working directory; nested imports are loaded relative to the importing module's lexical path.
- The supported import syntax is `from "path.frog" import name`, `from "path.frog" import name as alias`, and grouped whitespace-separated imports such as `from "path.frog" import ( x y z )`. Wildcards, commas, and `import "path.frog" as mod` are rejected for now.
- Relative import paths are resolved from the directory containing the importing module. Canonicalization is lexical and does not resolve symlinks.
- Imported files contribute procedures, records, unions, function-reference types, and macros. Imported top-level instructions are ignored and only the root module's `main` runs.
- Imported names are reexported, so facade modules can import a symbol and expose it to their importers.
- Macro declarations are collected with whole-module scope before the remaining code is compiled. Macro expansion is module-aware: imported and reexported macros resolve helper words in the module where the macro was defined. Recursive macro expansion is rejected.
- Every root program must define exactly one `proc main -- do ... end` with no inputs or outputs. Empty sources, declaration-only sources without `main`, and root top-level executable instructions are invalid.
- Typechecking occurs while procedures and expanded macros are compiled to C; failures include stack underflow, unknown words, contract mismatches, invalid control-flow stack shapes, and non-empty final stacks.
- Global `--debug` traces outer source-word type stacks to stderr during the measurement pass only. Optimized source words must still be traced without changing the measured slot bound or generated C bytes.
- Generated procedures use normal C parameters and return values. A measurement pass determines maximum operand depth, then emission uses scalar `Cell frog_value_N` locals for those positions; generated code has no runtime operand-stack object or per-procedure operand-stack array.
- Generated C emits source-requested headers, C-type assertions, and mechanical C-binding wrappers before compiler-private runtime headers. This ordering ensures a `c-call` or `c-value` cannot rely on declarations leaked by the runtime implementation.
- Generated C procedure names use `frog_proc_<global-id>_<encoded-source-name>`. ASCII letters and digits remain readable; every other UTF-8 byte, including `_`, is encoded as uppercase `_HH`. Numeric global IDs remain the function-reference dispatch identity.
- `compiler/frogc.c` must remain a checked fixed point: compiling `compiler/frogc.frog` with the seed and recompiling it with the result must reproduce the same C bytes.
- `bootstrap-update` compiles candidate compiler generations as standalone binaries and invokes their no-argument stdin-to-stdout filter mode.
- Bootstrap filter invocations run from `compiler/` so the compiler source's
  `../stdlib/libc.frog` import resolves the same way as a normal file build.

## Language Semantics

- User-facing language documentation lives in `docs/language.md`; update it when changing Frog syntax, semantics, intrinsics, examples, diagnostics that users see, or generated-C behavior.
- `macro name <body> end` records `<body>` as a compile-time token sequence in the defining module. Macro bodies may use function-body block constructs such as `if`, `while`, and `let`, but not nested `proc`, nested `macro`, or import declarations.
- `let a b c do ... end` binds visible stack values in source order: after `1 2 3`, `let a b c do` binds `a = 1`, `b = 2`, and `c = 3`. The implementation emits reverse-order pops to achieve this.
- `elif` is lowered to nested existing IF/ELSE/END instructions; one source `end` closes the whole chain, and the no-`else` path participates in stack-shape checking.
- The ordinary `read-file` procedure from `stdlib/libc.frog` consumes a UTF-8 path as `ptr int` and produces file bytes, byte length, and a success boolean as `ptr int bool`. On failure it returns zero length and `false`; the returned data pointer must not be dereferenced.
- `args` has stack effect `-- ptr int` and exposes the generated program's raw C `argv` followed by `argc`, including `argv[0]`; `@ptr` loads and `!ptr` stores one pointer-sized entry as `ptr`.
- `alloc`, `putc`, `getc`, `eputc`, and `exit` are ordinary procedures imported from `stdlib/libc.frog`, not language intrinsics.
- C interop declarations explicitly request system or local headers, name Frog-visible C types with trusted raw C type names, and bind calls or values. Calls retain fixed Frog arity even for variadic C declarations. Header declarations are authoritative: the compiler synthesizes neither C declarations nor dynamic loading.
- Shared libc/POSIX declarations live in `stdlib/libc.frog`; compiler and subprocess code import them instead of redeclaring private `cli-*` or `subprocess-*` aliases. Generated C wrappers perform only mechanical Cell/C-ABI conversion; wait/status and child-process policy stay in Frog.
- `record Name field Type ... end` defines a nominal pointer-backed record. `Name:alloc` allocates uninitialized storage, `Name:sizeof` exposes its Cell-based byte size, and `@Name.field`/`!Name.field` provide statically typed access.
- `@.field` and `!.field` infer the nominal record type from the top static stack value after macro expansion. Exact macros with those spellings take precedence; cover direct, macro-expanded, and imported-alias access when changing this behavior.
- Record fields occupy one eight-byte Cell in declaration order. Record-valued fields store handles, and only explicit `ptr`/record casts cross the nominal boundary.
- `union Name case Variant [PayloadType] ... end` defines a nominal pointer-backed tagged union with zero or one payload Cell per variant. `Name:variant` constructs, `Name.variant?` preserves and tests a validated handle, and `Name.variant` performs a checked projection.
- Union constructors allocate internal tag-and-payload storage. Union lifetime is manual, payload handles are borrowed, explicit `ptr`/union casts form an unsafe boundary, and matching through `if`/`elif` is not exhaustiveness-checked.
- `fn Name <inputs> -- <outputs> end` defines a nominal first-class function-reference contract. `Name:ref:procedure` creates an opaque one-Cell reference after exact contract checking, and `Name:call` consumes inputs followed by the reference and produces the declared outputs.
- Function calls dispatch only to generated procedure IDs with the exact resolved contract. Function references have no pointer/integer casts, allocation, lifetime, closure environment, anonymous syntax, implicit coercion, or C callback conversion.

## Implementation Conventions And Gotchas

- Keep language semantics and CLI policy in `compiler/frogc.frog`; generated-C runtime adapters should remain narrow ABI primitives rather than command parsers or build-policy implementations.
- When adding an intrinsic, update native recognition, type-stack behavior, emitted C/runtime support, bootstrap and regression coverage, user-facing docs, and optionally the VS Code grammar.
- Name direct compiler-internal pointer-field accessors `@object-field` and `!object-field`, used as `object @object-field` and `value object !object-field`. Keep indexed table operations and computed helpers under descriptive names instead of treating them as direct accessors.
- Compiler module state is the nominal `ModuleContext` record. Use generated `@ModuleContext.field` / `!ModuleContext.field` operations, keep semantic module values nominal, and confine raw casts to storage boundaries and null/identity checks; do not recreate manual `ctx-*` offset/accessor families.
- Fixed compiler metadata rows may use nominal records while their table bases remain contiguous raw allocations. Cast once in the row-address helper, size rows with `Type:sizeof`, and use generated typed field operations; `LocalEntry`, `ImportEntry`, `ScopeEntry`, and `BlockFrame` follow this pattern. `ConstantEvaluator` is a nominal record whose `values` field points to a separately grown raw value buffer.
- String literals lower to one `String` handle backed by a static byte-pointer/length descriptor. `String.bytes` and `String.len` expose its fields; byte storage is writable and shared by equal pooled literals, and generated globals and macro expansion must retain the defining module's pooled literal identity.
- Record and union type IDs share one program-global nominal allocator. Imported aliases and reexports retain the defining identity; type-level construction/allocation uses `:`, while fields and union instance operations use `.`.
- Exact macros may shadow generated nominal operations; otherwise record, union, and function operations resolve before locals and procedures with the same qualified spelling.
- Function-reference type IDs use a separate non-overlapping nominal range. Imported aliases and reexports retain the defining identity, and each generated indirect-call switch whitelists only exact-contract procedure IDs.
- The optimizer folds only adjacent integer literals followed by an unshadowed intrinsic `+`, and only after proving the sum fits signed 64-bit. Overflow behavior and exact macro precedence must remain unchanged.
- Generated scalar operand locals use `(void)&frog_value_N` to satisfy strict unused-variable diagnostics without reading an uninitialized value; `(void)frog_value_N` is not equivalent.
- Frog `int` is an `int64_t` cell in generated C. Fixed-width memory accesses must remain byte-safe through `memcpy` helpers.
- When adding a keyword, update native declaration/body scanning, macro validation, compilation, tests, docs, and `ide/vscode/frog_grammar.json`.
- User-facing compiler failures use stable `frogc: ...` diagnostics on standard error. Keep exact diagnostics covered by focused fixtures when practical.
- Static stack and operand failures caused by ordinary words use `frogc: <source-word>: <message>`. Preserve the outer source spelling across nested macro expansion.
- Contract mismatch diagnostics render expected and actual types in source order. Call failures render only the relevant actual stack suffix; function-reference failures render both full contracts. Qualify nominal types from imported modules with their defining canonical path.
- Keep ordinary language and compiler test declarations in Frog manifests. Python tests are reserved for host policies that Frog cannot conveniently control itself.
- Do not treat generated `.c` or `.exe` files as authoritative source, except for the intentional bootstrap seed `compiler/frogc.c`. Other generated files remain disposable build/test artifacts.
- CLI `build` intentionally writes outputs directly and provides no locking, rollback transaction, or path-alias validation. Users are responsible for choosing distinct input and output paths.

## VS Code Grammar

- The grammar is a small TextMate JSON package for `.frog` files.
- If language keywords, types, operators, word-like intrinsics, comments, or literals change, update `ide/vscode/frog_grammar.json` as part of the same change.
- The existing repository key is spelled `punctiation`; preserve or fix carefully because references currently use that spelling.

## Working Tree Hygiene

- The repository ignores generated `*.c`, `*.exe`, Python caches, mypy cache, `.devenv*`, `.direnv`, and local env files.
- `compiler/frogc.c` is the explicit exception to the generated-C ignore rule. Update the generated seed only with `just bootstrap-update`, whose fixed-point comparison must pass first.
- Before finalizing code changes, run `just test` when feasible. For docs-only changes, a lighter verification may be enough.
