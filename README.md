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

The compiler supports nominal pointer-backed records and tagged unions, nominal first-class function references, module-aware macros, an embedded standard macro prelude, scalar C foreign-function declarations, a deterministic checked literal-add peephole optimization, and root-relative imports of procedures, external functions, nominal types, and macros, including aliases, grouped imports, and reexports.

To bootstrap manually without Python:

```sh
mkdir -p build
gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 compiler/frogc.c -o build/frogc
build/frogc < compiler/frogc.frog > build/frogc.next.c
```

# VS Code extension

The dependency-free extension in `ide/vscode` provides syntax highlighting and associates `.frog` files with FrogLang. To install it from a source checkout, run these commands from the repository root on the machine where the VS Code extension host runs:

```sh
mkdir -p "$HOME/.vscode/extensions"
ln -s "$(pwd)/ide/vscode" "$HOME/.vscode/extensions/froglang-local"
```

Restart VS Code or run **Developer: Reload Window** from the command palette. Because the installation is a symbolic link, later changes in the same checkout become available after another reload.

To uninstall it:

```sh
unlink "$HOME/.vscode/extensions/froglang-local"
```

# Examples
-> [/examples/](./examples/)
