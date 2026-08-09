# FrogLang
FrogLang is a programming language:
- stack based
- concatenative
- statically typed
- compiled to C

It is heavily inspired by [Porth](https://gitlab.com/tsoding/porth)

# Usage

```sh
python -m frog --help
python -m frog run examples/01_simple.frog
```

`compiler/frogc.frog` is the sole Frog compiler, typechecker, and C emitter. The Python entrypoint contains only process and file orchestration: it invokes the checked compiler and `gcc`.

Every root program must define exactly one `proc main -- do ... end` with no inputs or outputs. Empty sources, declaration-only sources without `main`, and root top-level executable instructions are invalid.

`run` compiles Frog to temporary C, compiles that C to a temporary binary, and executes it. `build` locks both output paths and writes the generated `.c` and executable only after both compilation stages succeed; `build -r` holds those locks while running the published executable.

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

The compiler supports module-aware macros and root-relative imports of procedures and macros, including aliases, grouped imports, and reexports.

To bootstrap without Python:

```sh
gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 compiler/frogc.c -o frogc
./frogc < compiler/frogc.frog > frogc.next.c
```

# Examples
-> [/examples/](./examples/)
