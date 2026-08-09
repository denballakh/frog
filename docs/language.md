# FrogLang Language Reference

FrogLang is a small stack-based, concatenative, statically typed language. Programs use postfix stack operations, explicit stack-effect signatures, imports, macros, external C functions, and block keywords such as `proc`, `macro`, `if`, `else`, `while`, `do`, `end`, and `let`.

`compiler/frogc.frog` is the sole Frog compiler and typechecker. It emits C, which is compiled to the executable program; there is no separate interpreter or Python language implementation.

## Values And Literals

- Supported runtime value classes are `int`, `bool`, `ptr`, and `type`.
- Procedure signatures and casts can name `int`, `bool`, and `ptr`.
- `int` is a signed 64-bit integer in generated C. Integer literals are non-negative decimal, binary (`0b`), octal (`0o`), or hexadecimal (`0x`) chunks and must not exceed `9223372036854775807`. Base prefixes are lowercase; hexadecimal digits may be uppercase or lowercase. Negative values are produced by operations, not by signed literal syntax.
- `true` and `false` are bool literals.
- Character literals push integer codepoints.
- Character literals accept exactly one raw character. Backslash escape handling is not implemented.
- String literals push `ptr int`: a pointer to their bytes followed by their byte length. Their bytes are UTF-8 encoded; `\\`, `\"`, `\n`, `\r`, `\t`, `\0`, and `\xNN` escapes are supported. `\xNN` appends one byte. Double-quoted strings may span physical lines; raw line breaks and indentation inside the quotes are part of the value and are not normalized or stripped. Equal decoded byte strings share one generated storage object across all modules, even when their source spellings differ, so their pointers compare equal.
- Generated C string symbols use a deterministic hash of the decoded bytes and a collision suffix when unequal strings have the same hash. The generated C initializer has a trailing NUL byte, but Frog's explicit byte length excludes it and continues to preserve embedded NUL bytes.
- Import paths use string literal bytes decoded as UTF-8. Paths are limited to 1,024 decoded bytes and canonicalized lexically; symlinks are not resolved when determining module identity.
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

Procedure calls are statically checked against declared stack contracts and use the generated runtime cell stack.

## C Foreign Functions

External C functions use an explicit Frog name, C linker symbol, and scalar ABI contract:

```frog
extern magnitude abs c-int -- c-int end
extern release free c-ptr -- end

proc main -- do
    0 9 - magnitude print
    8 alloc release
end
```

The supported ABI types are `c-int` (Frog `int`, C `int`), `c-bool` (Frog `bool`, C `int` normalized to zero or one), and `c-ptr` (Frog `ptr`, C `void *`). An external function may consume any number of values and return zero or one value. It cannot be variadic.

The C symbol must be an ASCII C identifier that is not a C11 keyword or a generated-C name. The `frog_` prefix, numeric wrapper names such as `p0`, `main`, `Cell`, and `FrogStack` are reserved. Generated code declares and calls the symbol directly, so it must be provided by the C implementation or supplied when the generated C is linked. Frog does not load libraries dynamically or process C headers.

External functions use normal Frog name resolution and static stack-contract checking. They can be imported, aliased, and reexported like Frog procedures. Multiple Frog names may bind the same C symbol only when every declaration has the same ABI contract.

## Macros

Macros are compile-time token substitutions:

```frog
macro dup let x do x x end end
macro swap let x y do y x end end

proc main -- do
    1 2 swap drop drop
end
```

`macro name <body> end` records `<body>` as a token sequence. Macro declarations are collected before the remaining code is compiled, so macros have whole-file scope and can be used before or after their declaration. Whenever `name` appears as a word in the remaining code, it is expanded before normal word resolution, so macros can shadow intrinsics or procedures with the same name.

Macro bodies are syntax-checked for normal block structure and may use function-body constructs such as `if`, `while`, and `let`. `proc`, `extern`, and nested `macro` declarations are not valid inside a macro body. Recursive macro expansion is rejected.

## Imports

Imports make procedures, external functions, and macros from another Frog file visible in the importing module:

```frog
from "math.frog" import inc
from "math.frog" import inc as bump
from "math.frog" import ( inc dec add2 )

proc main -- do
    41 inc print
end
```

Only `from "path" import ...` is supported. Module alias imports such as `import "math.frog" as math` and wildcard imports are not supported. Grouped imports are whitespace-separated; commas are rejected.

Import declarations are collected before procedure bodies are compiled, so imported names can be used before the import declaration appears in the file.

Import paths are resolved relative to the root file being compiled, not relative to the importing module. For example, inside `pkg/use.frog`, `from "math.frog" import value` refers to the root-level `math.frog`; use `from "pkg/math.frog" import value` for the file under `pkg/`.

If the root source path is a symbolic link, imports are resolved from the lexical directory containing that link, not from the linked file's physical directory.

Imported files may reexport imported names:

```frog
// facade.frog
from "math.frog" import inc as bump
```

```frog
// main.frog
from "facade.frog" import bump

proc main -- do
    41 bump print
end
```

Imported top-level code is ignored. Imported files contribute procedure, external-function, and macro definitions, but only the root module's `main` runs.

Imported macros expand using the scope of the module where the macro was defined, even when reexported. Helper procedures and helper macros referenced by an imported macro are resolved in that defining module, not in the importing file.

Import cycles are rejected. Importing the same canonical file more than once is allowed, but two different symbols cannot be imported under the same visible name.

## Local Bindings

`let a b c do ... end` binds stack values to names in source order. If the stack is `1 2 3`, then `let a b c do` binds `a = 1`, `b = 2`, and `c = 3`.

Inside the implementation, bindings are emitted in reverse word order so the top of the stack is popped first, but the language-level behavior is source-order binding from the visible stack.

Example:

```frog
proc main -- do
    1 2 3
    let a b c do
        a print // 1
        b print // 2
        c print // 3
    end
end
```

## Control Flow

- `if <cond> do <then> [elif <cond> do <body> ...] [else <else>] end` requires every condition to preserve the stack from before `if` and add exactly one `bool`. Each arm, including the implicit no-op path when there is no `else`, must leave the same stack shape.
- `while <cond> do <body> end` requires the condition to preserve the stack from before `while` and add exactly one `bool`. The loop body must preserve the original loop stack shape.

## Language Constructs

- `proc name <inputs> -- <outputs> do ... end` defines a named procedure with an explicit stack-effect contract.
- `extern frog-name c-symbol <c-inputs> -- [c-output] end` declares a non-variadic C function with zero or one output.
- A root program must define exactly one explicit `proc main -- do ... end`; `main` cannot have inputs or outputs.
- Empty sources and declaration-only sources without `main` are invalid. Root top-level executable instructions are also invalid rather than being wrapped in a generated `main`.
- Only procedure, external-function, macro, and import declarations are allowed at the root top level. Imported top-level executable code is ignored.
- Procedure calls use the procedure name as a word and are statically checked against the declared contract.
- `macro name <body> end` defines a compile-time token substitution.
- `from "path" import name`, `from "path" import name as alias`, and `from "path" import ( name... )` import procedures, external functions, or macros from another file.
- `if ... do ... elif ... do ... else ... end` selects the first arm whose condition is true. `elif` may repeat; `else` is optional.
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

### Process Arguments

- `args`: `-- ptr int` pushes the raw C `argv` pointer followed by C `argc`. The count includes `argv[0]`.
- `argv` points to an array of C string pointers whose byte stride is the target C platform's pointer size. Use `@ptr` to load an entry; each resulting string is NUL-terminated and can be read with `@u8`.

### Memory

- `alloc`: `size_bytes -- ptr` allocates a byte buffer.
- `read-file`: `path_ptr path_length -- data_ptr data_length success` reads a UTF-8 path into an allocated byte buffer. On failure it returns length `0` and `false`; the data pointer must not be dereferenced.
- Pointer arithmetic supports `ptr int + -- ptr` and `ptr int - -- ptr`; offsets are in bytes.
- `int ptr +` is not supported.
- Signed pointer reads: `@i8`, `@i16`, `@i32`, `@i64`, each `ptr -- int`.
- Unsigned pointer reads: `@u8`, `@u16`, `@u32`, `@u64`, each `ptr -- int`.
- Pointer reads: `@ptr`, with stack effect `ptr -- ptr`; it copies one target-platform pointer-sized value.
- Signed pointer writes: `!i8`, `!i16`, `!i32`, `!i64`, each `int ptr --`.
- Unsigned pointer writes: `!u8`, `!u16`, `!u32`, `!u64`, each `int ptr --`.
- Generated C uses `malloc`, `void*`, byte pointer arithmetic, and fixed-width integer loads/stores from `<stdint.h>`.
- Generated C memory reads and writes use `memcpy`, so unaligned accesses do not violate C alignment or strict-aliasing rules.

### Casts

- `cast`: `x type -- y`
- Casts currently allow same-type, `int`/`bool`, `bool`/`int`, `int`/`ptr`, and `ptr`/`int` conversions.
- Casting `int` to `bool` produces `false` for zero and `true` for every nonzero value.
- The destination type is pushed with the `int`, `bool`, or `ptr` type word.

### Output And Debugging

- `print`: `a --`, prints one value with a newline.
- `putc`: `int --`, writes a single byte without an added newline. Generated C implements it using `putchar`.
- `getc`: `-- int`, reads one byte from standard input, or pushes `-1` at EOF.
- `eputc`: `int --`, writes one byte to standard error.
- `exit`: `int --`, terminates execution with the supplied exit status.
- `?`: `--`, a no-op debugging marker that is omitted from generated C.

## Generated C Limits

Frog uses signed 64-bit arithmetic in generated C. Signed overflow, division of `-9223372036854775808` by `-1`, and shifts with a negative or at-least-64 count are not defined. Right shift of negative values is implementation-defined in C. Pointer/integer casts use `intptr_t` and `uintptr_t`; they require a platform where object pointers fit in those types. An unsigned 64-bit read whose value exceeds the signed 64-bit range is implementation-defined when returned as Frog `int`. Passing a Frog `int` outside the C implementation's `int` range through `c-int` is also implementation-defined.
