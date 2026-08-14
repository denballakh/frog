# Formatter style

Frog code can use the stack state between physical lines as a layout rule. Code
flows downward, and indentation follows the result that one line leaves for the
next line. A physical line is one folded formatting unit: temporary stack
movement inside that line does not move it away from adjacent lines with the
same boundary state.

## Stack columns

Four spaces represent one syntax level. Two spaces represent one stack column.
Function bodies, constant expressions, conditions, control-flow arms, loop
bodies, and `let` or `peek` bodies are separate flow regions.

For one flow region, let `F` be the minimum stack depth at its entry or at a
boundary before or after one of its flow-owned physical lines. For a line, let
`B` be its stack depth before the line and `A` its stack depth after the line.
The line's column is:

```text
column = max(F, min(B, A - 1))
indent = 4 * syntax depth + 2 * (column - F)
```

If a line grows the stack, `A - 1` is at least `B`, so the line starts at `B`,
the first column it adds. If a line preserves or shrinks the stack, it follows
the top value it leaves at `A - 1`. A line that leaves no value above `F` stays
at the region's syntax baseline. Initializing `F` from the region entry prevents
ambient values below the region from shifting the whole region to the right.

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

A physical line of stack code is a folded flow unit. Analyze all of its tokens
to obtain the validated final depth `A`, but do not use intermediate depths for
indentation. This keeps implementation details such as a temporary conversion,
cleanup call, `swap`, or `dup` from producing a zigzag that is not present at
line boundaries.

For example, a folded `+ dup` line entered and exited at depth two follows the
top result column, even though `+` temporarily consumes both inputs:

```frog
a
  b
  + dup
```

Keeping tokens together hides their internal shape. These are expanded and
folded forms whose line-boundary columns remain the same:

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
grouping, normalize its indentation from the line's boundary depths, and never
reorder tokens. If a future formatter splits an overlong line, it must
recompute the boundary depths of every resulting line.

## Structured syntax

Stack columns apply to sequences of executable tokens. Declarations, imports,
struct and enum members, and `if`, `elif`, `else`, `while`, `let`, `peek`,
`do`, and `end` establish ordinary syntactic indentation and formatting
regions. Multiline conditions and bodies apply the stack-column rule within
their respective regions. If one physical line crosses regions, its first
token owns the line's syntax depth and region; the complete line still
determines its final boundary depth. A syntax token that starts a line keeps
ordinary syntax indentation.

Formatting requires the resolved stack effect of every source token it formats,
including overloaded function and macro invocation tokens. Literals and other
producers have effect `-- value`. Macro declarations are not formatted because
their effects can depend on expansion context; physical lines whose first code
token belongs to a macro declaration remain byte-for-byte unchanged. A
formatter should report an invalid stack program instead of guessing its
columns. Comments and literal contents must remain unchanged.

# Developer tooling

The compiler should expose reusable frontend, semantic-analysis, typed-IR, and
backend boundaries. Developer tools should consume the same source spans,
symbols, types, stack effects, and resolved operations as normal compilation
instead of implementing Frog semantics independently.

Possible tools and compiler capabilities include:

- a code optimizer operating on backend-independent typed IR;
- an LSP providing diagnostics, completion, hover information, definitions,
  references, symbols, rename support, and semantic tokens for syntax
  highlighting;
- the stack-aware formatter described above;
- a debugger that executes code interactively, steps by source operation, and
  exposes the current typed stack, locals, memory, and call stack;
- a REPL using the same execution engine as the debugger;
- an LLVM backend as an alternative to the C backend.

Additional useful tools enabled by the same compiler model:

- a stack and dataflow explainer showing the typed stack before and after each
  source word and the producer of every consumed value;
- compiler inspection commands for dumping tokens, resolved symbols, typed IR,
  control-flow graphs, and backend output;
- call graphs, module-dependency graphs, unused-symbol detection, and other
  semantic checks;
- stack-aware source refactoring, including safe rename and extraction of a
  selected flow region into a function with an inferred stack effect;
- test coverage and profiling reported in terms of Frog functions and source
  operations rather than generated C;
- API documentation generated from exported declarations, stack effects,
  structs, enums, C bindings, and source comments;
- a WebAssembly backend and browser playground, once backend-independent IR and
  execution are established.
