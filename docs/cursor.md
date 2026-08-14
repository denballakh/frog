# Semantic Cursor Queries

`frogc cursor` reports the symbols and typed operand stack available at one
byte position in a Frog source file. It performs the same loading, resolution,
and type analysis as `frogc check` before producing output.

Use a file or standard input:

```sh
build/frogc cursor --byte 42 examples/01_simple.frog
build/frogc cursor --byte 42 < examples/01_simple.frog
```

The offset is a nonnegative byte offset in the root source. It may equal the
source length to query end of file. An offset beyond the source length is a
command-line error. To query an imported file's source positions, invoke the
command with that file as the root.

If analysis fails, the command writes one diagnostic to standard error, exits
with status 1, and leaves standard output empty. Invalid command-line arguments
exit with status 2.

## Position selection

The query maps the byte offset to a token boundary:

- An offset inside a token selects the stack immediately before that token.
- An offset in whitespace or a comment selects the next token's boundary.
- An offset at end of file selects the boundary after the last token.

An occurrence is reported only when the requested byte is inside that
occurrence's half-open byte span. A trivia query can therefore have contexts,
visible symbols, and stacks without an occurrence row.

## Output format

The command emits a versioned tab-separated stream beginning with:

```text
frogc-cursor	2
```

Every `fields` row names the columns of the corresponding data row. Module,
scope, declaration, state, expansion, and local IDs belong to this analyzed
program. Consumers must check the format number; this is a diagnostic interface
for tests and developer-tool prototypes, not a stable serialization format.

The stream contains these row types:

- `query` identifies the root module, requested byte, and selected token.
- `context` identifies a semantic context at the selected boundary. A module
  context is always present. Function and constant contexts appear when the
  source position has retained semantic state. A macro-body position can have
  several function contexts because each caller stack produces a specialized
  expansion.
- `occurrence` describes each declaration, import binding, or reference whose
  source span contains the requested byte. Its semantic class distinguishes
  functions, macros, constants, locals, structs and fields, enums and cases,
  function-reference types, C types, primitive types, and intrinsics. Their
  exact semantic classes are `func`, `struct_type`, `struct_field`,
  `enum_type`, `enum_case`, and `function_type`; `function_type` remains
  specific to nominal `fn` declarations.
- `visible` lists lexically declared names available in a context. Rank `0` is
  a local, rank `1` is a module-scope name, and rank `2` is an implicit builtin.
  A local hides same-named outer locals, ordinary module names, and builtins.
  Module macros and nominal type words remain visible because Frog resolves
  them before locals. Primitive types and other syntax-provided words are not
  emitted as `visible` rows. Overload families produce multiple rows; a module
  function suppresses only a builtin overload with the same input types.
- `stack` gives the typed operand stack for a semantic context. Stack types use
  the same spelling as [`frogc inspect`](./inspect.md).

Declaration and reference identities use the canonical defining module and
declaration-table index. A struct field or enum case adds its member index; a
local uses its owning function identity and function-local ID. Imported
aliases keep that canonical identity and separately report the module and scope
index of the binding whose spelling was used. Missing identities or bindings
are written as `-1`.

Compound words have separate spans. For example, a query within `Point` in
`@Point.x` reports the struct identity, while a query within `x` reports the
field identity. Enum operations and function references split their owner and
member spans in the same way.

Use [`frogc inspect`](./inspect.md) when you need the complete analyzed program,
module paths, typed instructions, control-flow graph, or lossless source data.
