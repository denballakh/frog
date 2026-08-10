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

## Subprocesses

`stdlib/subprocess.frog` runs a program with captured standard output and
standard error:

```frog
from "../stdlib/subprocess.frog" import (
    CompletedProcess
    subprocess-argv
    subprocess-arg
    subprocess-argv-free
    subprocess-run
    completed-process-free
)

proc main -- do
    2 subprocess-argv
    let argv do
        "printf" argv 0 subprocess-arg
        "frog" argv 1 subprocess-arg

        argv "" subprocess-run
        let result do
            result @CompletedProcess.returncode print
            result @CompletedProcess.stdout_len print
            result completed-process-free
        end

        argv subprocess-argv-free
    end
end
```

- `subprocess-argv`: `count -- ptr` allocates a null-terminated argument array.
- `subprocess-arg`: `String argv index --` assigns one argument. Argument
  strings must not contain embedded NUL bytes, and the caller must use an index
  below the count supplied to `subprocess-argv`.
- `subprocess-argv-free`: `ptr --` releases the array. It does not release the
  string literals referenced by the array.
- `subprocess-run`: `argv input -- CompletedProcess` inherits the current
  directory and environment, supplies the `String` as standard input, waits for
  completion, and captures standard output and standard error separately.
- `subprocess-run-in`: `argv input cwd -- CompletedProcess` behaves the same way
  after changing the child to `cwd`. An empty `cwd` inherits the current
  directory.
- `completed-process-free`: `CompletedProcess --` releases both captured buffers
  and the result record.

`CompletedProcess.stdout` and `CompletedProcess.stderr` are byte pointers whose
lengths are stored in `stdout_len` and `stderr_len`; the buffers are not
NUL-terminated. `returncode` is the child exit status, or `128` plus the signal
number when the child is terminated by a signal. Commands are executed directly
without a shell, timeout, or environment rewriting.
