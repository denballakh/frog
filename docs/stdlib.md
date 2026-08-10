# Standard Library

Standard-library modules are ordinary Frog source files under `stdlib/`. Import
paths remain relative to the root source file, so a file in `examples/` can
import libc operations with:

```frog
from "../stdlib/libc.frog" import ( alloc free putc getc eputc exit )
```

## libc

`stdlib/libc.frog` exposes the small C-library surface used by Frog programs:

- `alloc`: `int -- ptr` allocates uninitialized bytes. The size must be
  non-negative and fit in the target C `int`.
- `free`: `ptr --` releases memory allocated by `alloc`.
- `putc`: `int --` writes the low byte to standard output.
- `getc`: `-- int` reads one byte from standard input, or returns `-1` at EOF.
- `eputc`: `int --` writes the low byte to standard error.
- `exit`: `int --` terminates the process with the supplied status.

The imported names are macros, so callers see the stack effects above rather
than the return values of the underlying C I/O functions.

## Strings And Byte Buffers

`stdlib/string.frog` provides operations for literal `String` values and
heap-backed byte buffers:

- `bytes-copy`: `source destination count --` copies `count` bytes.
- `bytes-equal`: `first first_len second second_len -- bool` compares byte
  ranges.
- `string-equal`: `String String -- bool` compares decoded string bytes.
- `byte-buffer-new`: `capacity -- ByteBuffer` creates an empty buffer.
- `byte-buffer-push`: `byte ByteBuffer --` appends one byte and grows the
  buffer when necessary.
- `byte-buffer-equal-string`: `ByteBuffer String -- bool` compares a buffer to
  a string.
- `byte-buffer-free`: `ByteBuffer --` releases a buffer and its storage.

`ByteBuffer` exposes `bytes`, `len`, and `capacity` fields through the normal
record operations. Its byte storage may move after `byte-buffer-push`, so code
must load `@ByteBuffer.bytes` again after an append.
