# FrogLang Language Reference

FrogLang is a small stack-based, concatenative, statically typed language. Programs use postfix stack operations, explicit stack-effect signatures, nominal records, tagged unions, first-class function references, imports, constants, macros, external C functions, and block keywords such as `proc`, `record`, `union`, `fn`, `const`, `macro`, `if`, `else`, `while`, `do`, `end`, `let`, and `peek`.

Compiler errors are written to standard error and start with `frogc:`. When an ordinary source word fails static stack or operand checks, the diagnostic includes that word's source spelling, for example `frogc: +: compile-time stack underflow`. A failure inside macro expansion is attributed to the outer source word that invoked the expansion.

Contract mismatch diagnostics show expected and actual types in brackets. Types are ordered from the lower stack position to the top, matching their order in source signatures. For a failed call, the actual list contains only the relevant top-of-stack suffix; unrelated values below it are omitted. Nominal types defined in imported modules use `<module-path>:<declaration-name>` so distinct same-named types remain distinguishable. Full function-reference and external C contracts show their input and output lists separated by `--`.

## Values And Literals

- Supported runtime value classes are `int`, `bool`, `ptr`, `String`, nominal record and union handles, nominal function references, and `type`.
- Procedure signatures can name `int`, `bool`, `ptr`, `String`, and visible record, union, or function-reference types.
- `int` is a signed 64-bit integer. Integer literals are non-negative decimal, binary (`0b`), octal (`0o`), or hexadecimal (`0x`) chunks and must not exceed `9223372036854775807`. Base prefixes are lowercase; hexadecimal digits may be uppercase or lowercase. Negative values are produced by operations, not by signed literal syntax.
- `true` and `false` are bool literals.
- Character literals push integer codepoints.
- Character literals accept exactly one raw character. Backslash escape handling is not implemented.
- String literals push one `String` value. `String.bytes` has stack effect `String -- ptr`, and `String.len` has stack effect `String -- int`. String bytes are UTF-8 encoded; `\\`, `\"`, `\n`, `\r`, `\t`, `\0`, and `\xNN` escapes are supported. `\xNN` appends one byte. Double-quoted strings may span physical lines; raw line breaks and indentation inside the quotes are part of the value and are not normalized or stripped.
- Equal decoded string literals share writable byte storage, so writes through `String.bytes` are visible through every equal literal in the program. The storage has a trailing NUL byte, while `String.len` excludes that terminator and includes embedded NUL bytes.
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

Procedure calls are statically checked against declared stack contracts.

## Records

Records define nominal reference values with typed fields:

```frog
record Node
    next Node
    value int
    ready bool
end

proc value-of Node -- int do @Node.value end

proc main -- do
    Node:alloc
    let node do
        41 node !Node.value
        true node !Node.ready
        node value-of print
    end
end
```

`record Name field Type ... end` is a top-level declaration. Record and field names are ASCII-style identifiers. Field types may be `int`, `bool`, `ptr`, `String`, or any visible nominal type. Record, union, and function-reference fields store handles rather than inline copies.

Record instances use manual memory management. `Name:alloc` has stack effect `-- Name` and allocates uninitialized storage for exactly that record. `Name:sizeof` has stack effect `-- int` and pushes the allocation size without allocating. `String` may be used as a field type but is a reserved built-in type, not a user-declarable record. There are no constructors, default field values, implicit allocation, ownership tracking, or garbage collection.

`@Name.field` reads a field with stack effect `Name -- FieldType`; `!Name.field` writes it with stack effect `FieldType Name --`. Type-level operations use `:`, union variants use `.`, and record access uses the familiar read/write sigils.

Record types are nominal. Two declarations with identical fields are different types, and field access requires the declared owner type. Explicit `ptr` to record and record to `ptr` casts are available for raw allocation and C FFI boundaries; direct casts between different record types are rejected.

See the runnable [records example](../examples/09_records.frog).

## Tagged Unions

Tagged unions define nominal alternatives with zero or one typed payload per variant:

```frog
union Result
    case ok int
    case error ptr
    case cancelled
end

proc main -- do
    42 Result:ok
    if Result.ok? do
        Result.ok print
    else
        Result.error drop
    end
end
```

`union Name case Variant [PayloadType] ... end` is a top-level declaration. Repeating `case` makes payloadless variants unambiguous without relying on line breaks. A union must declare at least one uniquely named variant. Payload types may be `int`, `bool`, `ptr`, `String`, or any visible nominal type.

`Name:variant` constructs a value, consuming the declared payload when present. `Name.variant?` validates the stored tag and has stack effect `Name -- Name bool`, preserving the handle so an immediately following `if` can project it. `Name.variant` validates that the value has exactly that variant, consumes the handle, and produces its payload; for a payloadless variant it only validates and consumes the handle. Invalid tags and wrong-variant projections terminate the program with status 1.

Union constructors allocate values; predicates and projections do not free them. Payload handles are borrowed, and unions do not own or recursively free their payloads. There is no uninitialized allocation or size operation for unions. Explicit `ptr`/union casts are available for manual lifetime and FFI boundaries; a pointer cast to a union must refer to a live value created by the matching union constructor.

Union types are nominal. Structurally identical declarations remain distinct, including through casts. Imported aliases and reexports retain the defining union's identity. Branching uses the existing `if`/`elif` constructs; matching is not exhaustiveness-checked.

An exact user-defined or imported macro may shadow a qualified record or union operation. Without such a macro, qualified nominal operations resolve before locals and procedures with the same spelling.

See the runnable [tagged-unions example](../examples/10_tagged_unions.frog).

## Function References

Named function-reference types describe a static Frog stack contract:

```frog
fn Mapper int -- int end

proc inc int -- int do 1 + end

proc apply int Mapper -- int do
    Mapper:call
end

proc main -- do
    41 Mapper:ref:inc apply print
end
```

`fn Name <inputs> -- <outputs> end` is a top-level declaration. `Name:ref:procedure` produces a `Name` reference only when `procedure` resolves to a visible Frog procedure with exactly the declared input and output counts and types. Forward references, imported procedure aliases, recursive procedures, and external Frog procedures are supported.

`Name:call` has stack effect `<inputs> Name -- <outputs>`: the function reference is on top of its inputs. Function-reference types are nominal, so independently declared types with identical contracts are not interchangeable. They may appear in procedure signatures, record fields, union payloads, and other function-reference contracts.

A function reference is opaque and can call only a procedure whose complete resolved contract matches the declared function-reference type. Function references cannot be cast to or from `int` or `ptr`, have no allocation or lifetime operations, and do not expose an underlying identity value.

There are no anonymous functions, captured environments, closures, implicit contract coercions, or C callback conversions. Exact macros may shadow `Name:ref:procedure` or `Name:call`; otherwise qualified function operations have the same precedence over locals and procedures as other nominal operations.

## C Foreign Functions

External C functions use an explicit Frog name, C linker symbol, and scalar ABI contract:

```frog
from "stdlib/libc.frog" import alloc

extern magnitude abs c-int -- c-int end
extern release free c-ptr -- end

proc main -- do
    0 9 - magnitude print
    8 alloc release
end
```

The supported ABI types are `c-int` (Frog `int`, C `int`), `c-bool` (Frog `bool`, C `int` normalized to zero or one), and `c-ptr` (Frog `ptr`, C `void *`). An external function may consume any number of values and return zero or one value. It cannot be variadic.

The C symbol must be an ASCII C identifier that is not a C11 keyword or a Frog-reserved name. The `frog_` prefix, `main`, `Cell`, and `FrogString` are reserved. The symbol must be available when the program is linked. Frog does not load libraries dynamically or process C headers.

External functions use normal Frog name resolution and static stack-contract checking. They can be imported, aliased, and reexported like Frog procedures. Multiple Frog names may bind the same C symbol only when every declaration has the same ABI contract.

The [C FFI example](../examples/11_c_ffi.frog) uses symbols from the C standard library. To provide symbols from another C source file, generate C and link both sources:

```sh
build/frogc < program.frog > program.c
gcc -std=c11 program.c helper.c -o program
```

## Macros

Macros are compile-time token substitutions:

```frog
macro dup let x do x x end end
macro swap let x y do y x end end

proc main -- do
    1 2 swap drop drop
end
```

`macro name <body> end` records `<body>` as a token sequence. Macro declarations are collected before the remaining code is compiled, so macros have whole-file scope and can be used before or after their declaration. User-defined and imported macros expand before normal word resolution, so they can shadow intrinsics or procedures with the same name.

Macro bodies are syntax-checked for normal block structure and may use function-body constructs such as `if`, `while`, and `let`. `proc`, `extern`, `record`, `union`, `fn`, `const`, and nested `macro` declarations are not valid inside a macro body. Recursive macro expansion is rejected.

## Compile-Time Constants

Constants evaluate a restricted postfix expression once during compilation and expand each use into the resulting typed literals:

```frog
const max-int 1 62 << 1 62 << 1 - + end
const answer-and-ready 6 7 * true end

proc main -- do
    max-int print
    answer-and-ready print print // true, then 42
end
```

`const name <expression> end` starts evaluation with an empty stack, infers the result arity and types, and requires at least one result. Results may be `int`, `bool`, or `String`; character literals produce `int`. Multiple results retain their bottom-to-top order. Evaluation happens once during compilation; each use pushes the stored results without reevaluating the expression at runtime.

Constant expressions accept literals, visible constant references, arithmetic and bitwise words (`+`, `-`, `*`, `/`, `%`, `/%`, `<<`, `>>`, `|`, `&`, `^`, `~`), boolean words (`&&`, `||`, `!`), and integer comparisons. They do not execute macros, procedures, control flow, local bindings, allocation, memory or I/O operations, casts, or nominal-type operations. Integer overflow, division by zero, and invalid shifts are compile errors. Constant shifts require a non-negative value and a count from 0 through 62.

Constants have whole-module scope, may refer forward to later constants, and are evaluated eagerly even when unused. Direct and indirect recursive definitions are rejected. Constants are importable, aliasable, and reexportable; their expressions resolve names in the module where they were defined. A macro may expand to a constant use, but macros are not executed inside constant definitions. Normal resolution prefers an exact macro, then types and intrinsics, then a local binding, then a constant or procedure, and finally a prelude macro.

## Standard Prelude

The standard prelude provides `dup`, `dup2`, `drop`, `swap`, `swap2`, and `rot` in every module. They behave like ordinary macros and may be shadowed.

Prelude names are fallback definitions. Resolution prefers a user-defined or imported macro, then a type or intrinsic, then a local binding, then a user-defined or imported constant or procedure, and finally a prelude macro. This permits any prelude word to be shadowed explicitly while keeping the standard names available in every module.

- `dup`: `a -- a a`
- `dup2`: `a b -- a b a b`
- `drop`: `a --`
- `swap`: `a b -- b a`
- `swap2`: `a b x y -- x y a b`
- `rot`: `a b c -- b c a`

## Imports

Imports make procedures, external functions, constants, records, unions, function-reference types, and macros from another Frog file visible in the importing module:

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

Relative import paths are resolved from the directory containing the importing module. For example, inside `pkg/use.frog`, `from "math.frog" import value` refers to `pkg/math.frog`, while `from "../math.frog" import value` refers to the root-level `math.frog`.

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

Imported top-level code is ignored. Imported files contribute procedure, external-function, constant, record, union, function-reference-type, and macro definitions, but only the root module's `main` runs. Imported nominal aliases retain the original identity and use the alias in qualified operations, such as `P:alloc`, `@P.value`, `M:some`, `M.some?`, and `F:call`.

Imported macros expand using the scope of the module where the macro was defined, even when reexported. Helper procedures and helper macros referenced by an imported macro are resolved in that defining module, not in the importing file.

Import cycles are rejected. Importing the same canonical file more than once is allowed, but two different symbols cannot be imported under the same visible name.

## Local Bindings

`let a b c do ... end` binds stack values to names in source order. If the stack is `1 2 3`, then `let a b c do` binds `a = 1`, `b = 2`, and `c = 3`.

`peek a b c do ... end` is equivalent to `let a b c do a b c ... end`. With distinct names, the captured values are restored in source order, so code in the block can inspect them without consuming the originals.

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

```frog
proc main -- do
    1 2
    peek a b do
        a b + print // 3
    end
    + print         // 3
end
```

## Control Flow

- `if <cond> do <then> [elif <cond> do <body> ...] [else <else>] end` requires every condition to preserve the stack from before `if` and add exactly one `bool`. Each arm, including the implicit no-op path when there is no `else`, must leave the same stack shape.
- `while <cond> do <body> end` requires the condition to preserve the stack from before `while` and add exactly one `bool`. The loop body must preserve the original loop stack shape.

## Language Constructs

- `proc name <inputs> -- <outputs> do ... end` defines a named procedure with an explicit stack-effect contract.
- `extern frog-name c-symbol <c-inputs> -- [c-output] end` declares a non-variadic C function with zero or one output.
- `record Name field Type ... end` defines a nominal record.
- `union Name case Variant [PayloadType] ... end` defines a nominal tagged union.
- `fn Name <inputs> -- <outputs> end` defines a nominal first-class function-reference contract.
- `const name <expression> end` eagerly evaluates a restricted expression and defines one or more typed literal values.
- A root program must define exactly one explicit `proc main -- do ... end`; `main` cannot have inputs or outputs.
- Empty sources and declaration-only sources without `main` are invalid. Root top-level executable instructions are also invalid; there is no implicit `main`.
- Only procedure, external-function, constant, record, union, function-reference, macro, and import declarations are allowed at the root top level. Imported top-level executable code is ignored.
- Procedure calls use the procedure name as a word and are statically checked against the declared contract.
- `macro name <body> end` defines a compile-time token substitution.
- `from "path" import name`, `from "path" import name as alias`, and `from "path" import ( name... )` import procedures, external functions, constants, records, unions, function-reference types, or macros from another file.
- `if ... do ... elif ... do ... else ... end` selects the first arm whose condition is true. `elif` may repeat; `else` is optional.
- `while ... do ... end` repeats while the condition leaves `true`.
- `let name... do ... end` binds visible stack values to local names in source order.
- `peek name... do ... end` binds visible stack values and evaluates those names before the block body.
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

### Process Arguments

- `args`: `-- ptr int` pushes the raw C `argv` pointer followed by C `argc`. The count includes `argv[0]`.
- `argv` points to an array of C string pointers whose byte stride is the target C platform's pointer size. Use `@ptr` to load an entry; each resulting string is NUL-terminated and can be read with `@u8`.

### Memory

- `read-file`: `path_ptr path_length -- data_ptr data_length success` reads a UTF-8 path into an allocated byte buffer. On failure it returns length `0` and `false`; the data pointer must not be dereferenced.
- Pointer arithmetic supports `ptr int + -- ptr` and `ptr int - -- ptr`; offsets are in bytes.
- `int ptr +` is not supported.
- Signed pointer reads: `@i8`, `@i16`, `@i32`, `@i64`, each `ptr -- int`.
- Unsigned pointer reads: `@u8`, `@u16`, `@u32`, `@u64`, each `ptr -- int`.
- Pointer reads: `@ptr`, with stack effect `ptr -- ptr`; it copies one target-platform pointer-sized value.
- Pointer writes: `!ptr`, with stack effect `ptr ptr --`; it copies the first pointer value into the address on top of the stack.
- Signed pointer writes: `!i8`, `!i16`, `!i32`, `!i64`, each `int ptr --`.
- Unsigned pointer writes: `!u8`, `!u16`, `!u32`, `!u64`, each `int ptr --`.
- Memory reads and writes support unaligned addresses.

### Casts

- `cast`: `x type -- y`
- Casts allow same-type, `int`/`bool`, `bool`/`int`, `int`/`ptr`, `ptr`/`int`, and `ptr`/record-or-union-handle conversions. `String` and function-reference types support only same-type casts.
- Casting `int` to `bool` produces `false` for zero and `true` for every nonzero value.
- The destination type is pushed with the `int`, `bool`, `ptr`, `String`, or visible record, union, or function-reference type word.

### Output And Debugging

- `print`: `int --` or `bool --`, prints one value with a newline.
- `?`: `--`, a no-op debugging marker.

Byte allocation, byte-oriented standard I/O, memory release, and process termination are provided by [`stdlib/libc.frog`](stdlib.md#libc), not by the language.

## Runtime Limits

Runtime signed overflow, division of `-9223372036854775808` by `-1`, and shifts with a negative or at-least-64 count have unspecified results. Right shift of a negative value is platform-dependent. Compile-time constant arithmetic rejects these cases instead. Pointer/integer casts require a target where object pointers fit in an integer. An unsigned 64-bit read above `9223372036854775807`, or passing a Frog `int` outside the target C `int` range through `c-int`, is platform-dependent.
