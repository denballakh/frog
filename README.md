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
build/frogc check examples/01_simple.frog
build/frogc fmt examples/01_simple.frog
build/frogc inspect examples/01_simple.frog
build/frogc cursor --byte 0 examples/01_simple.frog
build/frogc run examples/01_simple.frog
```

Every module permits declarations only at the top level. Every root program must define exactly one `func main -- do ... end` with no inputs or outputs; empty and declaration-only root sources are invalid.

## Command-line interface

`build/frogc -h` shows CLI help. With no arguments, `build/frogc` is a compiler filter: it reads Frog source from standard input and writes generated C to standard output. `check` performs full semantic analysis without generating C; it reads the named file, or standard input when no file is given. `fmt` accepts input the same way and writes stack-aware indentation to standard output without changing the source file or adding line breaks. See the [formatter reference](./docs/formatter.md). `inspect` prints the analyzed program for developer tools. Pass `--builtins` to include implicitly imported builtin function bodies. Pass `--lexemes` to include lossless source lexemes and syntax regions; both options may be combined. `cursor --byte OFFSET [FILE]` reports semantic identities, visible names, and typed stacks at a root-source byte position. These versioned outputs are diagnostic interfaces and may change between versions. See the [compiler inspection reference](./docs/inspect.md) and [semantic cursor reference](./docs/cursor.md). Imported module paths are relative to the root source directory. `run` writes reusable scratch artifacts under `build/`, compiles them, and executes the binary. `build` writes the source-adjacent `.c` and executable directly; `build -r` runs the resulting executable.

Prefix a filter, `check`, `fmt`, `inspect`, `cursor`, `run`, or `build` invocation with `--debug` to trace compile-time type stacks to standard error. Use `--release` to omit calls to the implicit builtin `assert` while retaining operand evaluation. See the [language reference](./docs/language.md) for details.

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
- [Stack-aware formatter](./docs/formatter.md)
- [Compiler inspection output](./docs/inspect.md)
- [Semantic cursor queries](./docs/cursor.md)
- [Testing and bootstrap verification](./docs/testing.md)
- [VS Code support](./docs/ide/vscode.md)
- [Examples](./examples/README.md)

The complete documentation index is available in [`docs/README.md`](./docs/README.md).
