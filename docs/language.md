# FrogLang Language Reference

FrogLang is a small stack-based, concatenative, statically typed language. Programs use postfix stack operations, explicit stack-effect signatures, nominal structs, enums, first-class function references, imports, constants, macros, C interop declarations, and block keywords such as `func`, `struct`, `enum`, `fn`, `const`, `macro`, `if`, `else`, `while`, `do`, `end`, `let`, and `peek`.

## Contents

- [Diagnostics](#diagnostics)
- [Values and literals](#values-and-literals)
- [Stack effects](#stack-effects)
- [Functions](#functions)
- [Structs](#structs)
- [Enums](#enums)
- [Function references](#function-references)
- [C interop](#c-interop)
- [Macros](#macros)
- [Compile-time constants](#compile-time-constants)
- [Implicit builtins](#implicit-builtins-module)
- [Compiler intrinsics](#compiler-intrinsics)
- [Imports](#imports)
- [Local bindings](#local-bindings)
- [Control flow](#control-flow)
- [Operators and memory](#operators)
- [Runtime limits](#runtime-limits)

## Diagnostics

Compiler errors are written to standard error. Source-located errors use this
format:

```text
PATH:LINE:COLUMN:
  SOURCE-LINE
  CARETS
error: MESSAGE
```

Compilation and analysis stop after the first diagnostic. The `check`,
`inspect`, and `cursor` commands exit with status 1 and leave standard output
empty when analysis fails.

`PATH` is `<stdin>` for filter input, `<command>` for `run -c`, and the loaded
module path for files. Columns are visual columns from one; tabs are expanded
to 8-column stops when the source line and carets are displayed. A
multi-line token highlights its first source line. When an ordinary source word
fails static stack or operand checks, the message identifies that word, for
example `error: +: compile-time stack underflow`. A failure inside macro
expansion is attributed to the outer source word that invoked the expansion.

Contract mismatch diagnostics show expected and actual types in brackets. Types are ordered from the lower stack position to the top, matching their order in source signatures. For a failed call, the actual list contains only the relevant top-of-stack suffix; unrelated values below it are omitted. Nominal types defined in imported modules use `<module-path>:<declaration-name>` so distinct same-named types remain distinguishable. Full function-reference and C-call contracts show their input and output lists separated by `--`.

## Values and literals

- Supported runtime value classes are `int`, the exact-width integer types, `bool`, `ptr`, typed pointers, `String`, nominal struct and enum values, nominal function references, and `type`.
- Function, struct-field, enum-payload, and function-reference signatures can name `int`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`, `ptr`, `String`, and visible nominal types.
- `int` is a target-sized signed integer. Integer literals are decimal, binary (`0b`), octal (`0o`), or hexadecimal (`0x`) chunks with an optional leading `+` or `-`. Literals outside the target `int` range are unsupported. Base prefixes are lowercase; hexadecimal digits may be uppercase or lowercase. A standalone `+` or `-` remains an arithmetic word.
- `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, and `u64` are distinct static integer types. Integer literals are still `int`, so conversion to or between exact-width types requires `cast`; no implicit integer conversion is performed.
- A trailing `*` names a typed pointer: `i8*`, `String*`, `Node*`, and `Node**` are valid. Pointer types are structural and canonicalized by their pointee, so repeated spelling and imported aliases of the same nominal type yield the same pointer type. Different pointee types remain distinct. `ptr` is the untyped raw-pointer boundary; casts are available between `ptr` and a typed pointer, but no implicit conversion is performed.
- `true` and `false` are bool literals.
- Character literals push one `int` codepoint. They contain exactly one raw
  Unicode codepoint or one of `\\`, `\'`, `\"`, `\n`, `\r`, `\t`, `\0`, and
  `\xNN`, where `NN` is exactly two hexadecimal digits. Other escapes and a raw
  backslash are invalid; write the backslash character as `'\\'`.
- String literals push a `String` value, a copied descriptor containing a byte pointer and length. `String.bytes` has stack effect `String -- ptr`, and `String.len` has stack effect `String -- int`. String bytes are UTF-8 encoded; `\\`, `\"`, `\n`, `\r`, `\t`, `\0`, and `\xNN` escapes are supported. `\xNN` appends one byte. Double-quoted strings may span physical lines; raw line breaks and indentation inside the quotes are part of the value and are not normalized or stripped.
- Copying a `String` copies its descriptor, not its bytes. Equal decoded string literals share writable byte storage, so writes through `String.bytes` are visible through every equal literal in the program. The storage has a trailing NUL byte, while `String.len` excludes that terminator and includes embedded NUL bytes.
- Import paths use string literal bytes decoded as UTF-8. Paths are limited to 1,024 decoded bytes and canonicalized lexically; symlinks are not resolved when determining module identity.
- `//` starts a line comment only when tokenized as its own whitespace-delimited chunk.

## Stack effects

Stack effects are written with inputs before `--` and outputs after it. For example, `int int -- int` consumes two integers and produces one integer.

The rightmost stack item is the top of the stack. For example, after `1 2 3`, the stack is `1 2 3`, with `3` on top.

`frogc --debug` traces the complete compile-time type stack immediately before and after each outer source word. The trace is written to standard error, lists types from the bottom of the stack to the top, and does not change generated C or program output. Function calls, intrinsics, macros, local and constant references, type words, and nominal operations are traced. Literals, declarations, control keywords, `let`, and `peek` are not separate trace entries. A macro reports its aggregate effect under the caller's spelling; its expansion details are not exposed. If a word fails, its `before` entry precedes the normal diagnostic and there is no `after` entry.

`frogc --release` omits calls resolved implicitly to the builtin `assert`, but
still evaluates their operands. Explicitly imported, aliased, or shadowing
assertions continue to run. Global `--debug` and `--release` options may appear
in either order before a subcommand or before filter input.

## Functions

Functions declared with `func` use explicit stack-effect signatures:

```frog
func inc int -- int do
    1 +
end
```

Function calls are statically checked against declared stack contracts.

A root program must define exactly one `main` function with the contract
`--`. Empty files and declaration-only root files without `main` are invalid.
In every module, top-level source is limited to
function, C interop, constant, struct, enum, function-reference, macro, and
import declarations. Standalone executable instructions at the top level are
rejected.

## Structs

Structs define nominal value types with typed fields; `Name*` is the corresponding typed pointer type:

```frog
struct Node
    next Node*
    value int
    ready bool
end

func value-of Node -- int do @Node.value end

func main -- do
    true 41 Node:new !Node.value !Node.ready value-of print
end
```

`struct Name field Type ... end` is a top-level declaration. Struct and field names are ASCII-style identifiers. Field types may be primitive, typed-pointer, or visible nominal types. Struct, enum, and function-reference fields store their values directly. A recursive inline struct or enum field is rejected; use a typed pointer such as `Name*` to make a recursive edge.

`Name:new` has stack effect `-- Name` and produces a zero-initialized value. `Name:alloc` has stack effect `-- Name*` and allocates a zero-initialized pointed-to value. `Name:sizeof` has stack effect `-- int` and pushes the size of the value without allocating. `String` may be used as a field type but is a reserved built-in type, not a user-declarable struct. There are no default field values, implicit allocation, ownership tracking, or garbage collection.

`@Name.field` reads a field with stack effects `Name -- FieldType` and `Name* -- FieldType`. `!Name.field` is a functional setter for a value (`FieldType Name -- Name`) and a mutating setter for a pointer (`FieldType Name* --`). `@.field` and `!.field` infer `Name` from the value or pointer on top of the static stack after macro expansion, with the same respective stack effects. Exact macros named `@.field` or `!.field` take precedence over inferred access. Type-level operations use `:`, enum variants use `.`, and struct access uses the familiar read/write sigils.

Struct value and pointer types are nominal. Two declarations with identical fields are different types, and field access requires a value or pointer to the declared owner type. Explicit `ptr` to struct-pointer and struct-pointer to `ptr` casts are available for raw allocation and C interop boundaries; direct casts between different struct pointer types are rejected.

See the runnable [structs example](../examples/09_structs.frog).

## Enums

Enums define nominal alternatives with zero or one typed payload per variant:

```frog
enum Result
    case ok int
    case error ptr
    case cancelled
end

func main -- do
    42 Result:ok
    if dup Result.ok? do
        Result.ok print
    else
        Result.error drop
    end
end
```

`enum Name case Variant [PayloadType] ... end` is a top-level declaration. Repeating `case` makes payloadless variants unambiguous without relying on line breaks. An enum must declare at least one uniquely named variant. Payload types may be primitive, typed-pointer, or visible nominal types.

`Name:variant` constructs a value, copying the declared payload when present. `Name.variant?` validates the stored tag, consumes the value, and produces a boolean (`Name -- bool`). Use `dup Name.variant?` when both the value and the predicate result are needed. `Name.variant` validates that the value has exactly that variant, consumes the value, and produces its payload; for a payloadless variant it only validates and consumes the value. Invalid tags and wrong-variant projections terminate the program with status 1.

Enum values and their payloads pass and copy by value. There is no allocation, ownership tracking, or size operation for enums.

Enum types are nominal. Structurally identical declarations remain distinct, including through casts. Imported aliases and reexports retain the defining enum's identity. Branching uses the existing `if`/`elif` constructs; matching is not exhaustiveness-checked.

An exact user-defined or imported macro may shadow a qualified struct or enum operation. Without such a macro, qualified nominal operations resolve before locals and functions with the same spelling.

See the runnable [tagged-enums example](../examples/10_tagged_enums.frog).

## Function references

Named function-reference types describe a static Frog stack contract:

```frog
fn Mapper int -- int end

func inc int -- int do 1 + end

func apply int Mapper -- int do
    Mapper:call
end

func main -- do
    41 Mapper:ref:inc apply print
end
```

`fn Name <inputs> -- <outputs> end` declares a nominal function-reference type; it does not declare an executable function. `Name:ref:function` produces a `Name` reference only when `function` resolves to a visible Frog function with exactly the declared input and output counts and types. Forward references, imported function aliases, recursive functions, and `c-call` functions are supported.

Functions may be overloaded by their complete ordered input contract. Their outputs do not distinguish overloads, so two functions with the same name and inputs are a duplicate even if their outputs differ. A call selects the single visible overload whose inputs exactly match the top of the static stack; no match is a contract mismatch and multiple matches are ambiguous. Imports, aliases, and reexports preserve the complete overload family. User and imported overloads are considered before builtin functions, but an unmatched user overload does not hide a matching builtin signature.

`Name:call` has stack effect `<inputs> Name -- <outputs>`: the function reference is on top of its inputs. Function-reference types are nominal, so independently declared types with identical contracts are not interchangeable. They may appear in function signatures, struct fields, enum payloads, and other function-reference contracts.

A function reference is opaque and can call only a function whose complete resolved contract matches the declared function-reference type. Function references cannot be cast to or from `int` or `ptr`, have no allocation or lifetime operations, and do not expose an underlying identity value.

There are no anonymous functions, captured environments, closures, implicit contract coercions, or C callback conversions. Exact macros may shadow `Name:ref:function` or `Name:call`; otherwise qualified function-reference operations have the same precedence over locals and ordinary functions as other nominal operations.

## C interop

C interop declarations explicitly name the headers, C type spellings, calls, and values that a Frog module uses:

```frog
c-include system "stdlib.h" end
c-include system "string.h" end

c-type CInt int "int" end
c-type CSize int "size_t" end
c-type CBytes ptr "const char *" end
c-type CPtr ptr "void *" end

c-call magnitude abs CInt -- CInt end
c-call length strlen CBytes -- CSize end
c-call allocate malloc CSize -- CPtr end
c-call release free CPtr -- end

func main -- do
    -9 magnitude print
    "frog" String.bytes length print
    8 allocate release
end
```

`c-include system "name.h" end` emits a system-header include. `c-include local "path.h" end` emits a local-header include. Every referenced C function or object must be declared, and every referenced C macro must be defined, by an explicitly included header. Frog does not synthesize declarations or load libraries dynamically.

`c-type Name int|bool|ptr "C type name" end` maps a trusted C type name to a Frog representation. For example, `"size_t"` maps `size_t` to Frog `int`, `"FILE *"` maps `FILE *` to Frog `ptr`, and `"void (*)(int)"` may be represented as Frog `ptr`. `int` values use ordinary C integer conversion, `bool` results are normalized to `true` or `false`, and `ptr` values use the target's object-pointer/integer representation.

`c-call FrogName CSymbol Inputs -- [Output] end` binds a C function or function-like macro as a Frog function. Its Frog contract has fixed arity, even when the header declaration is variadic. `c-value FrogName CSymbol -- Output end` binds a C object or object-like macro as a Frog function. Both kinds of binding can be imported, aliased, reexported, and used as function-reference targets. C types can likewise be imported and reexported, including under aliases.

The C symbol must be an ASCII C identifier that is not a C11 keyword or a Frog-reserved name. The `frog_` prefix, `main`, `Cell`, and `FrogString` are reserved. The [C interop example](../examples/11_c_ffi.frog) uses standard-library headers. A separately linked helper needs a local header that declares the functions or objects it exposes:

```sh
build/frogc < program.frog > program.c
gcc -std=c11 program.c helper.c -o program
```

Function-pointer types such as `"void (*)(int)"` can be represented as Frog `ptr`, but passing or storing those values uses target-dependent function-pointer/integer conversion. This is an unsafe boundary; portable Frog code should avoid function-pointer interop.

## Macros

Macros are compile-time token substitutions:

```frog
macro dup let x do x x end end
macro swap let x y do y x end end

func main -- do
    1 2 swap drop drop
end
```

`macro name <body> end` records `<body>` as a token sequence. Macro declarations are collected before the remaining code is compiled, so macros have whole-file scope and can be used before or after their declaration. User-defined and imported macros expand before normal word resolution, so they can shadow intrinsics or functions with the same name.

Macro bodies are syntax-checked for normal block structure and may use function-body constructs such as `if`, `while`, and `let`. `func`, `c-include`, `c-type`, `c-call`, `c-value`, `struct`, `enum`, `fn`, `const`, and nested `macro` declarations are not valid inside a macro body. Recursive macro expansion is rejected.

## Compile-time constants

Constants evaluate a restricted postfix expression once during compilation and expand each use into the resulting typed literals:

```frog
const max-int 1 62 u32 cast << 1 62 u32 cast << 1 - + end
const answer-and-ready 6 7 * true end

func main -- do
    max-int print
    answer-and-ready print print // true, then 42
end
```

`const name <expression> end` starts evaluation with an empty stack, infers the result arity and types, and requires at least one result. Results may be integer types, `bool`, or `String`; character literals produce `int`. Multiple results retain their bottom-to-top order. Evaluation happens once during compilation; each use pushes the stored results without reevaluating the expression at runtime.

Constant expressions accept literals, visible constant references, explicit integer casts, arithmetic and bitwise words (`+`, `-`, `*`, `/`, `%`, `/%`, `<<`, `>>`, `|`, `&`, `^`, `~`), boolean words (`&&`, `||`, `!`), and integer comparisons. Operators require the same exact contracts as runtime operators; in particular, shift counts are `u32`. They do not execute macros, other functions, control flow, local bindings, allocation, memory or I/O operations, or nominal-type operations. Overflow checks apply to `int` constant arithmetic; exact-width arithmetic follows its C-style result conversion. Division by zero and invalid `int` shifts are compile errors.

Constants have whole-module scope, may refer forward to later constants, and are evaluated eagerly even when unused. Direct and indirect recursive definitions are rejected. Constants are importable, aliasable, and reexportable; their expressions resolve names in the module where they were defined. A macro may expand to a constant use, but macros are not executed inside constant definitions. Normal resolution prefers an exact macro, then types and intrinsics, then a local binding, then a constant or function, and finally a builtins definition.

## Implicit builtins module

The compiler loads [`stdlib/builtins.frog`](../stdlib/builtins.frog) from Frog source for every program. Its `dup`, `dup2`, `drop`, `swap`, `swap2`, `rot`, `NULL`, and `assert` definitions are available without an import in every other module. The stack operations are macros; `NULL` and `assert` are ordinary functions.

Builtins are fallback definitions. Resolution prefers a user-defined or imported macro, then a type or intrinsic, then a local binding, then a user-defined or imported constant or matching function overload, and finally a matching builtins definition. This permits any builtin word to be shadowed. The module may also be imported explicitly, for example with `from "stdlib/builtins.frog" import assert`.

- `dup`: `a -- a a`
- `dup2`: `a b -- a b a b`
- `drop`: `a --`
- `swap`: `a b -- b a`
- `swap2`: `a b x y -- x y a b`
- `rot`: `a b c -- b c a`
- `NULL`: `-- ptr`; produces the null untyped pointer. Test pointers explicitly with `value NULL ==` or `value NULL !=`.
- `assert`: `bool String --`; a true condition does nothing. A false condition writes the message followed by a newline to standard error and terminates the program with status 1.

In release mode, calls resolved implicitly to this builtin `assert` consume their operands without invoking the assertion function. Operand expressions are still evaluated in source order. User-defined assertions, explicit imports or aliases of the builtin function, and function-reference calls retain their normal behavior.

## Compiler intrinsics

Words beginning with `__intrinsic_` are low-level compiler operations. They can
be called in function bodies in any module and are resolved before local,
constant, function, or builtin words. Their names remain reserved: functions,
macros, constants, nominal types, C bindings, and import aliases cannot declare
names with this prefix.

The standard operators use typed intrinsics such as `__intrinsic_add_int` and
`__intrinsic_load_i8`. Direct calls have the same exact stack contracts as the
corresponding operations and bypass shadowable builtin functions.
`__intrinsic_assert_fail` has stack effect `String --`; it writes the string and
a newline to standard error, then terminates the program with status 1.

## Imports

Imports make functions, C calls and values, C types, constants, structs, enums, function-reference types, and macros from another Frog file visible in the importing module:

```frog
from "math.frog" import inc
from "math.frog" import inc as bump
from "math.frog" import ( inc dec add2 )

func main -- do
    41 inc print
end
```

Only `from "path" import ...` is supported. Module alias imports such as `import "math.frog" as math` and wildcard imports are not supported. Grouped imports are whitespace-separated; commas are rejected.

Import declarations are collected before function bodies are compiled, so imported names can be used before the import declaration appears in the file.

Import paths beginning with `stdlib/` are resolved from the compiler
distribution's standard-library root. They do not fall back to the importing
module's directory. Prefix the path with `./` to import a local directory named
`stdlib` instead. Because `stdlib/builtins.frog` is loaded for every program,
the compiler distribution's standard-library root must always be available.

All other relative import paths are resolved from the directory containing the
importing module. For example, inside `pkg/use.frog`,
`from "math.frog" import value` and `from "./math.frog" import value` refer to
`pkg/math.frog`, while `from "../math.frog" import value` refers to the
root-level `math.frog`.

If the root source path is a symbolic link, imports are resolved from the lexical directory containing that link, not from the linked file's physical directory.

Imported files may reexport imported names:

```frog
// facade.frog
from "math.frog" import inc as bump
```

```frog
// main.frog
from "facade.frog" import bump

func main -- do
    41 bump print
end
```

Imported files must contain declarations only at the top level. They contribute function, C call, C value, C type, constant, struct, enum, function-reference-type, and macro declarations, but only the root module's `main` runs. Imported nominal aliases retain the original identity and use the alias in qualified operations, such as `P:alloc`, `@P.value`, `@.value`, `M:some`, `M.some?`, and `F:call`.

Imported macros expand using the scope of the module where the macro was defined, even when reexported. Helper functions and helper macros referenced by an imported macro are resolved in that defining module, not in the importing file.

Import cycles are rejected. Importing the same canonical file more than once is allowed, but two different symbols cannot be imported under the same visible name.

## Local bindings

`let a b c do ... end` binds stack values to names in source order. If the stack is `1 2 3`, then `let a b c do` binds `a = 1`, `b = 2`, and `c = 3`.

`peek a b c do ... end` is equivalent to `let a b c do a b c ... end`. With distinct names, the captured values are restored in source order, so code in the block can inspect them without consuming the originals.

Example:

```frog
func main -- do
    1 2 3
    let a b c do
        a print // 1
        b print // 2
        c print // 3
    end
end
```

```frog
func main -- do
    1 2
    peek a b do
        a b + print // 3
    end
    + print         // 3
end
```

## Control flow

- `if <cond> do <then> [elif <cond> do <body> ...] [else <else>] end` requires every condition to preserve the stack from before `if` and add exactly one `bool`. Each arm, including the implicit no-op path when there is no `else`, must leave the same stack shape.
- `while <cond> do <body> end` requires the condition to preserve the stack from before `while` and add exactly one `bool`. The loop body must preserve the original loop stack shape.

## Operators

Operators are overloaded implicit builtins. They have no special source-level
namespace and follow ordinary visible-word shadowing rules. Exact-width operands
must match exactly: Frog never widens or narrows them implicitly.

### Arithmetic

- `+`, `-`, `*`, `/`, `%`, `/%`: matching `int` or matching exact integer widths; `+` and `-` also support `ptr int -- ptr`.

### Bitwise

- `<<`, `>>`: `int u32 -- int` and matching exact-width integer left operands with `u32` shift counts.
- `|`, `&`, `^`, `~`: matching `int` or exact integer widths.

### Logic

- `&&`: `bool bool -- bool`
- `||`: `bool bool -- bool`
- `!`: `bool -- bool`

### Comparisons

- `==`, `!=`, `<`, `>`, `<=`, `>=`: matching `int` or exact integer widths. Pointers support `==` and `!=`.

### Process arguments

- `args`: `-- ptr int` pushes the raw C `argv` pointer followed by C `argc`. The count includes `argv[0]`.
- `argv` points to an array of C string pointers whose byte stride is the target C platform's pointer size. Use `ptr* cast @` to load an entry; each resulting string is NUL-terminated and can be read with `u8* cast @`.

### Memory

- Pointer arithmetic supports `ptr int + -- ptr` and `ptr int - -- ptr`; offsets are in bytes.
- `int ptr +` is not supported.
- `@` reads one value from a typed pointer: `T* -- T`. `!` writes one value: `T T* --`. For example, `address u8* cast @ int cast` reads a byte for use as an `int`, and `value u8 cast address u8* cast !` writes a byte.
- The width-spelled memory words (`@u8`, `!i32`, `@ptr`, and similar) are removed.
- Memory reads and writes support unaligned addresses.

### Casts

- `cast`: `x type -- y`
- Casts allow same-type, conversions among `int` and the exact-width integer types, `int`/`bool`, `bool`/`int`, `int`/`ptr`, `ptr`/`int`, and `ptr`/typed-pointer. `String`, struct and enum values, and function-reference types support only same-type casts.
- Casting `int` to `bool` produces `false` for zero and `true` for every nonzero value.
- The destination type is pushed with a primitive, typed-pointer, or visible nominal type word.

### Output and debugging

- `print`: `int --` or `bool --`, prints one value with a newline.
- `?`: `--`, a no-op debugging marker.

Byte allocation, byte-oriented standard I/O, memory release, and process termination are provided by [`stdlib/libc.frog`](stdlib.md#libc), not by the language.

## Runtime limits

Arithmetic operators use the corresponding C integer operations after type
selection. Runtime signed overflow, division of the minimum signed value by
`-1`, and shifts with a count greater than or equal to the left operand's width
have unspecified results. Right shift of a negative value is platform-dependent.
Compile-time `int` constant arithmetic rejects these cases; exact-width constant
arithmetic follows its C-style result conversion. Pointer/integer casts require
a target where object pointers fit in an integer. C interop conversions follow
the declared C type, so values outside that target C type's range and
function-pointer values represented as `ptr` are target-dependent.
