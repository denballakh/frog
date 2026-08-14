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
Procedure bodies, constant expressions, conditions, control-flow arms, loop
bodies, and `let` or `peek` bodies are separate flow regions.

For each region, `F` is the minimum stack depth at the region entry or at a
boundary before or after one of its flow-owned physical lines. For one line,
`B` is its stack depth before the line and `A` is its depth after the line. The
formatter computes:

```text
column = max(F, min(B, A - 1))
indent = 4 * syntax depth + 2 * (column - F)
```

If a line grows the stack, it aligns with the first column it adds. If it
preserves or shrinks the stack, it aligns with the top value it leaves. A line
that leaves no value above `F` stays at the syntax baseline. Temporary pushes,
consumption, and shuffles inside a folded line do not affect its indentation.

This makes lines that preserve a carried value align even if they create and
discard temporary values:

```frog
entry @JsonObjectEntry.next
entry @JsonObjectEntry.key json-text-free
entry @JsonObjectEntry.value json-free
entry ptr cast free
```

If one physical line crosses syntax or flow regions, its first token owns the
line's syntax depth and region. The complete line still determines its final
stack depth. This allows compact control flow to remain folded:

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
- physical lines whose first code token belongs to a macro declaration;
- LF, CRLF, and lone-CR line endings.

The formatter does not wrap long lines or split, join, or reorder tokens.
Macro declarations are preserved instead of formatted because their stack
effects can depend on the types at each expansion site.

Formatting requires one resolved stack effect for every executable source
token it visits. Macro invocations inside procedures are formatted from the
concrete net effect resolved while analyzing that procedure. Invalid source or
an invalid loaded dependency produces a diagnostic instead of formatted
output.
