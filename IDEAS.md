# Formatter style

Frog code can use its stack effects as a layout rule. Code flows downward. A
value that continues through several operations forms a vertical spine, while
additional operands are branches above and to the right of the operation that
consumes them.

## Stack columns

For a token evaluated at stack depth `d` with stack effect `i -- o`, its stack
column is the lower boundary of the stack slice that it replaces:

```text
column = d - i
new depth = d - i + o
```

Stack columns are zero-based boundaries between stack depths. Immediately
before the token, its inputs occupy `[d - i, d)`. Immediately afterward, its
outputs occupy `[d - i, d - i + o)`. A producer with effect `0 -- 1` at depth
`d` therefore places its result in column `d`. A `1 -- 2` word keeps its lower
output in the consumed input's column and places its other output one column to
the right.

Therefore:

- `0 -- 1` starts a branch in the next free column;
- `1 -- 1` continues a column;
- `2 -- 1` joins two columns at the lower one;
- `1 -- 0` terminates a column;
- `1 -- 2` splits a column;
- multi-input and multi-output words occupy the lowest column they touch.

Two spaces represent one stack column. Syntax nesting, such as a procedure or
an `if` body, continues to use four spaces. Each surrounding code region has a
shared stack floor: the lowest boundary touched by any executable token in the
region, including values consumed from the region's entry stack. A physical
line is placed using:

```text
visual stack column = line floor - region floor
indent = syntax indent + 2 * visual stack column
```

The common region floor preserves the relative positions of all lines while
preventing values below the region from shifting the entire region to the
right. Procedure and macro bodies, constant expressions, conditions, control
flow arms, loop bodies, and `let` or `peek` bodies are separate regions.

For example, fully expanding `a b + c *` produces:

```frog
a
  b
+
  c
*
```

The same rule formats a nested computation as branches above its consumers:

```frog
const json-int-max
    1
      62 u32 cast
    <<
      1
        62 u32 cast
      <<
        1
      -
    +
end
```

## Tokens grouped on one line

A physical line of stack code is a folded flow unit. Treat all of its tokens
as one compound stack program and align the line with the lowest stack column
touched anywhere in that program.

More precisely, let `D0` be the depth before the line. For every token `t`, let
`d(t)` be its input depth and `i(t)` its input count. The line floor is:

```text
floor = min(D0, d(t) - i(t) for every token t on the line)
```

If the depth after the line is `D1`, the composed effect of the line is:

```text
(D0 - floor) -- (D1 - floor)
```

Thus treating a line as one compound operation and using its lowest stack
floor are equivalent, provided the compound effect is composed from every
token. Merely using `D1 - D0` is wrong because it loses temporary consumption.
For example, `+ dup` has no net depth change, but its composed effect is
`2 -- 2` and it is anchored at the column of its lower input.

Keeping tokens together folds their internal shape without changing the outer
layout. These are the expanded and folded forms of the same flow:

```frog
a
  b
    c
  +
*
```

```frog
a
  b c +
*
```

Authors may keep short computations, conversions, loads, calls, and linear
pipelines on one line. Splitting a line exposes more of its internal flow;
joining adjacent tokens folds it. A formatter should preserve this deliberate
grouping initially, normalize its indentation from the composed effect, and
never reorder tokens. If a future formatter splits an overlong line, it must
recompute the floor of every resulting line.

## Structured syntax

Stack columns apply to sequences of executable tokens. Declarations, imports,
record and union members, and `if`, `elif`, `else`, `while`, `let`, `peek`,
`do`, and `end` establish ordinary syntactic indentation and formatting
regions. Multiline conditions and bodies apply the stack-column rule within
their respective regions.

Formatting requires the resolved stack effect of every source token, including
overloaded procedures and macros. Literals and other producers have effect
`-- value`. A formatter should report an unresolved or invalid stack program
instead of guessing its columns. Comments and literal contents must remain
unchanged.
