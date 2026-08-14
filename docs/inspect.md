# Compiler Inspection Output

`frogc inspect` exposes the compiler's analyzed program for tests and developer
tools. It performs the same loading, name resolution, and type analysis as
`frogc check`, then writes a versioned tab-separated stream to standard output.
If analysis fails, it writes one diagnostic to standard error, exits with
status 1, and leaves standard output empty.

Use a file or standard input:

```sh
build/frogc inspect examples/01_simple.frog
build/frogc inspect < examples/01_simple.frog
```

The stream starts with `frogc-inspect`, a tab, and its format number. `fields`
rows name the columns that follow. Module and declaration indexes are local to
the inspected program. Imported paths are relative to the root source
directory when possible.

This interface is intended for compiler tests and early developer-tool
prototypes. Consumers must check the format number; it is not a stable public
serialization format.

For a position-oriented view of resolved identities, visible names, and the
typed stack, use [`frogc cursor`](./cursor.md).

## Format 4: analyzed program

Ordinary `inspect` emits format 4. Its rows describe:

- loaded modules and procedures;
- resolved source operations and their typed stack before and after execution;
- typed backend-independent instructions;
- caller-specialized macro expansion origins;
- local bindings; and
- typed control-flow blocks and terminators.

Pass `--builtins` to include procedure bodies from the implicitly imported
builtins module. The builtins module itself remains listed without this option
so that resolved targets retain a visible module identity.

## Format 5: lossless source data

Pass `--lexemes` to emit format 5. It contains every format-4 row plus `lexeme`
and `region` rows:

```sh
build/frogc inspect --lexemes examples/01_simple.frog
```

Each lexeme has a module index, lexeme index, byte start, byte length, kind,
and escaped text. Lexemes are a lossless ordered partition of the source:
tokens, comments, and whitespace together cover every source byte exactly
once.

Text escapes tab, newline, carriage return, backslash, and non-printable bytes
as `\t`, `\n`, `\r`, `\\`, and `\xHH`. Lexeme text renders each ordinary ASCII
space as `<space>` so trailing or whitespace-only lexemes remain visible.

Each syntax region has a module-local ID, parent ID, kind, half-open token
range, and half-open byte range. The module region has parent `-1` and covers
the complete source, including leading and trailing trivia. Declaration
regions cover their declaration tokens. Block regions identify nested `if`,
`while`, `let`, and `peek` constructs inside procedure and macro declarations.

Combine `--lexemes` with `--builtins` in either order to include lossless source
data and analyzed bodies for the builtins module:

```sh
build/frogc inspect --builtins --lexemes examples/01_simple.frog
```
