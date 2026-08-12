# FrogLang

FrogLang is a programming language that is:

- stack based
- concatenative
- statically typed
- compiled to C

## Requirements

Building FrogLang requires a POSIX environment, [just](https://just.systems/),
and a C11-capable GCC installation. The repository's devenv configuration
provides these tools together with the Python formatting and type-checking
tools used by the test suite.

## Quick start

```sh
just frogc-seed
build/frogc -h
build/frogc run examples/01_simple.frog
```

Every root program must define exactly one `proc main -- do ... end` with no inputs or outputs. Empty sources, declaration-only sources without `main`, and root top-level executable instructions are invalid.

## Command-line interface

`build/frogc -h` shows CLI help. With no arguments, `build/frogc` is a compiler filter: it reads Frog source from standard input and writes generated C to standard output. `run` writes reusable scratch artifacts under `build/`, compiles them, and executes the binary. `build` writes the source-adjacent `.c` and executable directly; `build -r` runs the resulting executable.

Prefix a filter, `run`, or `build` invocation with `--debug` to trace compile-time type stacks to standard error. Use `--release` to omit calls to the implicit builtin `assert` while retaining operand evaluation. See the [language reference](./docs/language.md) for details.

## Bootstrap

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

To bootstrap manually without Python:

```sh
mkdir -p build
gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 compiler/frogc.c -o build/frogc
build/frogc < compiler/frogc.frog > build/frogc.next.c
```

## Documentation

- [Language reference](./docs/language.md)
- [Standard library](./docs/stdlib.md)
- [Testing and bootstrap verification](./docs/testing.md)
- [VS Code support](./docs/ide/vscode.md)
- [Examples](./examples/README.md)

The complete documentation index is available in [`docs/README.md`](./docs/README.md).
