# FrogLang Language Reference

FrogLang is a small stack-based, concatenative, statically typed language. Programs use postfix stack operations, explicit stack-effect procedure signatures, and block keywords such as `proc`, `if`, `else`, `while`, `do`, `end`, and `let`.

## Compiler Pipeline

- `tokenize(text, filename)` creates tokens with source positions.
- `compile(toks)` lowers tokens into an IR containing procedures and instructions.
- Top-level instructions create an implicit `main` procedure; an empty program still gets an empty `main`.
- Explicit `proc main -- do ... end` must have no inputs and no outputs.
- `typecheck(ir)` simulates stack effects and rejects stack underflows, unknown words, wrong contracts, bad branch/loop stack states, and non-empty final stacks.
- `interpret(ir)` executes Frog directly.
- `translate(ir)` emits C code, and the CLI can compile it with `gcc`.

## Values And Literals

- Supported runtime value classes are `int`, `bool`, `ptr`, and `type`.
- Procedure signatures and casts can name `int`, `bool`, and `ptr`.
- Integer literals are non-negative decimal chunks. Negative values are produced by operations, not by signed literal syntax.
- `true` and `false` are bool literals.
- Character literals push integer codepoints and are supported by both interpretation and C codegen.
- Character literals accept exactly one raw character. Backslash escape handling is not implemented.
- String literals tokenize, but compilation currently reports `not implemented: string literals`.
- `//` starts a line comment only when tokenized as its own whitespace-delimited chunk.

## Stack Effects

Stack effects are written with inputs before `--` and outputs after it. For example, `int int -- int` consumes two integers and produces one integer.

The rightmost stack item is the top of the stack. For example, after `1 2 3`, the stack is `1 2 3`, with `3` on top.

## Procedures

Procedures use explicit stack-effect signatures:

```frog
proc inc int -- int do
    1 +
end
```

Procedure calls are statically checked against declared stack contracts. Return values are modeled as struct fields in generated C.

## Local Bindings

`let a b c do ... end` binds stack values to names in source order. If the stack is `1 2 3`, then `let a b c do` binds `a = 1`, `b = 2`, and `c = 3`.

Inside the implementation, bindings are emitted in reverse word order so the top of the stack is popped first, but the language-level behavior is source-order binding from the visible stack.

Example:

```frog
1 2 3
let a b c do
    a print // 1
    b print // 2
    c print // 3
end
```

## Control Flow

- `if <cond> do <then> [else <else>] end` requires the condition to leave exactly one bool and both branches to leave the same stack shape.
- `while <cond> do <body> end` requires the condition to leave exactly one bool and the loop body to preserve the original stack shape.

## Intrinsics

Intrinsics include arithmetic, bitwise, logic, comparisons, stack manipulation, memory operations, `cast`, `print`, `putc`, and `?` debug.

### Memory

- `alloc`: `size_bytes -- ptr` allocates a byte buffer.
- Pointer arithmetic supports `ptr int + -- ptr` and `ptr int - -- ptr`; offsets are in bytes.
- `int ptr +` is not supported.
- Typed pointer reads use `ptr @i<n> -- int` and `ptr @u<n> -- int` for signed/unsigned `n`-bit values.
- Typed pointer writes use `val ptr !i<n> --` and `val ptr !u<n> --` for signed/unsigned `n`-bit values.
- Supported memory access widths are `8`, `16`, `32`, and `64`.
- The interpreter models allocated memory as bytearray-backed pointers and checks bounds/fit for memory access.
- Generated C uses `malloc`, `void*`, byte pointer arithmetic, and fixed-width integer loads/stores from `<stdint.h>`.

### Casts

Casts currently allow same-type, `int`/`bool`, `bool`/`int`, `int`/`ptr`, and `ptr`/`int` conversions.

### Output And Debugging

- `print` consumes one value and prints it with a newline.
- The interpreter prints `[PRINT] ...`; generated C prints raw values without the `[PRINT]` prefix.
- `putc` consumes an `int` codepoint and writes a single character without an added newline or interpreter prefix.
- Generated C implements `putc` using `putchar`.
- `?` logs the stack at compile time during typechecking and at runtime during interpretation; it is omitted in C codegen.
