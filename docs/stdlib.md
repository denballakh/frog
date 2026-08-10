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
