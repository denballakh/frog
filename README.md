# FrogLang
FrogLang is a programming language:
- stack based
- concatenative
- statically typed
- compiled to C

It is heavily inspired by [Porth](https://gitlab.com/tsoding/porth)

# Usage

```sh
just frogc-seed
build/frogc -h
build/frogc run examples/01_simple.frog
```

`compiler/frogc.frog` implements the compiler, typechecker, C emitter, and CLI. Python is test-only.

Every root program must define exactly one `proc main -- do ... end` with no inputs or outputs. Empty sources, declaration-only sources without `main`, and root top-level executable instructions are invalid.

`build/frogc -h` shows CLI help. With no arguments, `build/frogc` is a compiler filter: it reads Frog source from standard input and writes generated C to standard output. `run` writes reusable scratch artifacts under `build/`, compiles them, and executes the binary. `build` writes the source-adjacent `.c` and executable directly; `build -r` runs the resulting executable.

Prefix a filter, `run`, or `build` invocation with `--debug` to trace compile-time type stacks to standard error. Debug tracing does not change generated C or program output.

# Compiler and bootstrap

The checked fixed-point C bootstrap seed is `compiler/frogc.c`, so a C compiler is sufficient to bootstrap Frog:

```sh
just frogc-seed
build/frogc < compiler/frogc.frog > build/frogc.next.c
cmp compiler/frogc.c build/frogc.next.c
```

Run the complete two-generation fixed-point verification with:

```sh
just bootstrap-check
```

The language supports strings, nominal records and tagged unions, first-class function references, compile-time constants and macros, C interop declarations, and source-relative imports with aliases, groups, and reexports. See the [language reference](./docs/language.md) for syntax and semantics.

To bootstrap manually without Python:

```sh
mkdir -p build
gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 compiler/frogc.c -o build/frogc
build/frogc < compiler/frogc.frog > build/frogc.next.c
```

# Editor support

VS Code extension features, installation, updates, and removal are documented in [`docs/ide/vscode.md`](./docs/ide/vscode.md).

# Examples
-> [/examples/](./examples/)

Feature-focused examples:

- [records](./examples/09_records.frog)
- [tagged unions](./examples/10_tagged_unions.frog)
- [C interop](./examples/11_c_ffi.frog)
