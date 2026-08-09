# FrogLang
FrogLang is a programming language:
- stack based
- concatenative
- statically typed
- compiled or interpreted

It is heavily inspired by [Porth](https://gitlab.com/tsoding/porth)

# Usage
```sh
py -m frog --help
```

# Self-hosting compiler

The compiler source is `compiler/frogc.frog`. Its fixed-point C seed is checked in so a C compiler is sufficient to bootstrap Frog:

```sh
just frogc-seed
build/frogc < compiler/frogc.frog > build/frogc.next.c
cmp compiler/frogc.c build/frogc.next.c
```

Run the complete two-generation fixed-point verification with:

```sh
just bootstrap-check
```

The self-hosted compiler currently requires an explicit `proc main -- do ... end`. Macros and imports are still being migrated from the Python compiler.

# Examples
-> [/examples/](./examples/)
