# Stack-aware formatter

`frogc fmt` normalizes Frog indentation from the analyzed stack effects while
preserving each physical line and the author's token grouping.

## Format source

Pass one file, or omit the file to read standard input:

```sh
build/frogc fmt examples/01_simple.frog
build/frogc fmt < examples/01_simple.frog
```

Formatted source is written to standard output. The command never changes the
input file. To replace a file, write to a separate path and move it only after
the command succeeds.

The formatter performs full syntax and semantic analysis before writing any
source. Invalid input produces a diagnostic on standard error, exits with
status 1, and leaves standard output empty. A root module may contain only
declarations and does not need a `main` procedure. File input resolves relative
imports from the input file's directory.

## Indentation rule

Four spaces represent one syntax level. Two spaces represent one stack column.
For an operation reached at stack depth `d` with effect `i -- o`, the operation
touches stack floor `d - i` and leaves depth `d - i + o`.

Each flow region has a shared floor: the lowest boundary touched by any
operation in the region. Procedure and macro bodies, constant expressions,
conditions, control-flow arms, loop bodies, and `let` or `peek` bodies are
separate regions. A flow line is indented as:

```text
4 * syntax depth + 2 * (line floor - region floor)
```

All operations grouped on one physical line within the owning flow region
contribute to that line's floor. This includes temporary stack consumption, so
a folded line such as `+ dup` is aligned from its complete `2 -- 2` effect, not
from its zero net depth change.

If one physical line crosses syntax or flow regions, its first token owns the
line's indentation. Later regions cannot reclassify or move that line. This
allows compact control flow to remain folded:

```frog
proc choose -- int do
    2 3 + if dup 5 == do drop 7 else drop 11 end
end
```

Splitting operations across lines exposes their stack shape. For example,
`a b + c *` formats as:

```frog
    a
      b
    +
      c
    *
```

## Preserved source

The formatter changes only leading spaces and tabs on physical lines whose
first source token is code. It preserves:

- the number and order of physical lines;
- all tokens and their intra-line spacing;
- comments, blank lines, and trailing whitespace;
- literal bytes, including multiline literal continuation lines;
- LF, CRLF, and lone-CR line endings.

The formatter does not wrap long lines or split, join, or reorder tokens.

Formatting requires one resolved stack effect for every executable source
token it visits. This includes overloads and macro bodies. If an unused macro
body has no analyzed specialization, formatting reports `formatter requires a
resolved stack effect` instead of guessing its indentation. If callers resolve
one macro-body token to different effects, formatting reports conflicting
resolved stack effects.
