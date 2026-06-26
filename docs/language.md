# FrogLang Language Reference

FrogLang is a small stack-based, concatenative, statically typed language. Programs use postfix stack operations, explicit stack-effect procedure signatures, imports, macros, and block keywords such as `proc`, `macro`, `if`, `else`, `while`, `do`, `end`, and `let`.

## Values And Literals

- Supported runtime value classes are `int`, `bool`, `ptr`, and `type`.
- Procedure signatures and casts can name `int`, `bool`, and `ptr`.
- Integer literals are non-negative decimal chunks. Negative values are produced by operations, not by signed literal syntax.
- `true` and `false` are bool literals.
- Character literals push integer codepoints and are supported by both interpretation and C codegen.
- Character literals accept exactly one raw character. Backslash escape handling is not implemented.
- String literals are supported in import declarations. In normal code, string literals tokenize, but compilation currently reports `not implemented: string literals`.
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

## Macros

Macros are compile-time token substitutions:

```frog
macro dup let x do x x end end
macro swap let x y do y x end end

1 2 swap
```

`macro name <body> end` records `<body>` as a token sequence. Macro declarations are collected before the remaining code is compiled, so macros have whole-file scope and can be used before or after their declaration. Whenever `name` appears as a word in the remaining code, it is expanded before normal word resolution, so macros can shadow intrinsics or procedures with the same name.

Macro bodies are syntax-checked for normal block structure and may use function-body constructs such as `if`, `while`, and `let`. `proc` and nested `macro` definitions are not valid inside a macro body. Recursive macro expansion is rejected.

## Imports

Imports make procedures and macros from another Frog file visible in the importing module:

```frog
from "math.frog" import inc
from "math.frog" import inc as bump
from "math.frog" import ( inc dec add2 )

41 inc print
```

Only `from "path" import ...` is supported. Module alias imports such as `import "math.frog" as math` and wildcard imports are not supported. Grouped imports are whitespace-separated; commas are rejected.

Import declarations are collected before procedure bodies and top-level code are compiled, so imported names can be used before the import declaration appears in the file.

Import paths are resolved relative to the root file being compiled, not relative to the importing module. For example, inside `pkg/use.frog`, `from "math.frog" import value` refers to the root-level `math.frog`; use `from "pkg/math.frog" import value` for the file under `pkg/`.

Imported files may reexport imported names:

```frog
// facade.frog
from "math.frog" import inc as bump
```

```frog
// main.frog
from "facade.frog" import bump
```

Imported top-level code is ignored. Imported files contribute procedure and macro definitions, but only the root module's `main` runs.

Imported macros expand using the scope of the module where the macro was defined, even when reexported. Helper procedures and helper macros referenced by an imported macro are resolved in that defining module, not in the importing file.

Import cycles are rejected. Importing the same canonical file more than once is allowed, but two different symbols cannot be imported under the same visible name.

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

- `if <cond> do <then> [else <else>] end` requires the condition to preserve the stack from before `if` and add exactly one `bool`. The `do` consumes that bool. Both branches must leave the same stack shape.
- `while <cond> do <body> end` requires the condition to preserve the stack from before `while` and add exactly one `bool`. The loop body must preserve the original loop stack shape.

## Language Constructs

- `proc name <inputs> -- <outputs> do ... end` defines a named procedure with an explicit stack-effect contract.
- Top-level instructions are compiled into an implicit `main` procedure.
- `proc main -- do ... end` may be defined explicitly, but it must have no inputs and no outputs.
- Procedure calls use the procedure name as a word and are statically checked against the declared contract.
- `macro name <body> end` defines a compile-time token substitution.
- `from "path" import name`, `from "path" import name as alias`, and `from "path" import ( name... )` import procedures or macros from another file.
- `if ... do ... else ... end` selects one of two branches. `else` is optional.
- `while ... do ... end` repeats while the condition leaves `true`.
- `let name... do ... end` binds visible stack values to local names in source order.
- `//` starts a line comment only when it appears as its own whitespace-delimited token.

## Intrinsics

### Arithmetic

- `+`: `int int -- int`, `ptr int -- ptr`
- `-`: `int int -- int`, `ptr int -- ptr`
- `*`: `int int -- int`
- `/`: `int int -- int`
- `%`: `int int -- int`
- `/%`: `int int -- int int`, producing quotient then remainder

### Bitwise

- `<<`: `int int -- int`
- `>>`: `int int -- int`
- `|`: `int int -- int`
- `&`: `int int -- int`
- `^`: `int int -- int`
- `~`: `int -- int`

### Logic

- `&&`: `bool bool -- bool`
- `||`: `bool bool -- bool`
- `!`: `bool -- bool`

### Comparisons

- `==`: `int int -- bool`
- `!=`: `int int -- bool`
- `<`: `int int -- bool`
- `>`: `int int -- bool`
- `<=`: `int int -- bool`
- `>=`: `int int -- bool`

### Stack Manipulation

- `dup`: `a -- a a`
- `dup2`: `a b -- a b a b`
- `drop`: `a --`
- `swap`: `a b -- b a`
- `swap2`: `a b x y -- x y a b`
- `rot`: `a b c -- b c a`

### Memory

- `alloc`: `size_bytes -- ptr` allocates a byte buffer.
- Pointer arithmetic supports `ptr int + -- ptr` and `ptr int - -- ptr`; offsets are in bytes.
- `int ptr +` is not supported.
- Signed pointer reads: `@i8`, `@i16`, `@i32`, `@i64`, each `ptr -- int`.
- Unsigned pointer reads: `@u8`, `@u16`, `@u32`, `@u64`, each `ptr -- int`.
- Signed pointer writes: `!i8`, `!i16`, `!i32`, `!i64`, each `int ptr --`.
- Unsigned pointer writes: `!u8`, `!u16`, `!u32`, `!u64`, each `int ptr --`.
- The interpreter models allocated memory as bytearray-backed pointers and checks bounds/fit for memory access.
- Generated C uses `malloc`, `void*`, byte pointer arithmetic, and fixed-width integer loads/stores from `<stdint.h>`.

### Casts

- `cast`: `x type -- y`
- Casts currently allow same-type, `int`/`bool`, `bool`/`int`, `int`/`ptr`, and `ptr`/`int` conversions.
- The destination type is pushed with the `int`, `bool`, or `ptr` type word.

### Output And Debugging

- `print`: `a --`, prints one value with a newline.
- `putc`: `int --`, writes a single character without an added newline or interpreter prefix. Generated C implements `putc` using `putchar`.
- `?`: `--`, logs the stack at compile time during typechecking and at runtime during interpretation; it is omitted in C codegen.
