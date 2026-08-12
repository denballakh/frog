# Standard Library

Standard-library modules are ordinary Frog source files imported through the
`stdlib/` module search path:

```frog
from "stdlib/libc.frog" import ( alloc free putc getc eputc exit )
```

The compiler locates this search path from its own executable path. Invoke it
with a path containing `/`, such as `build/frogc`; a name found only through
the shell's `PATH` does not identify the accompanying standard library.

## builtins

`stdlib/builtins.frog` is an ordinary Frog module loaded implicitly for every
program. It defines the standard stack macros `dup`, `dup2`, `drop`, `swap`,
`swap2`, and `rot`, plus the `assert` procedure documented in the
[language reference](language.md#implicit-builtins-module). Its definitions are
available without an import and may be shadowed; explicit imports from the
module are also supported.

## libc

`stdlib/libc.frog` declares its C dependencies with `c-include`, maps trusted header
types with `c-type`, and wraps header-declared functions, macros, and values
with private `c-call` and `c-value` bindings. The compiler emits no C
declarations for those symbols.

It exposes this small C-library surface to Frog programs:

- `alloc`: `int -- ptr` allocates uninitialized bytes. The size must be
  non-negative and representable as the target C `size_t`.
- `free`: `ptr --` releases memory allocated by `alloc`.
- `putc`: `int --` writes the low byte to standard output.
- `getc`: `-- int` reads one byte from standard input, or returns `-1` at EOF.
- `eputc`: `int --` writes the low byte to standard error.
- `exit`: `int --` terminates the process with the supplied status.
- `read-file`: `path_ptr path_length -- data_ptr data_length success` reads a
  path into an allocated byte buffer. On failure it returns length `0` and
  `false`; the data pointer must not be dereferenced. On success the caller
  releases the data with `free`.

These are ordinary Frog procedures, so callers use the Frog contracts above
rather than the underlying C signatures.

The module also exposes the POSIX operations used by the compiler and
subprocess library:

- `fork`: `-- int`
- `create-file`: `path -- int`
- `dup2`: `old_fd new_fd -- int`
- `close`: `fd -- int`
- `chdir`: `path -- int`
- `get-current-directory`: `buffer capacity -- success` writes the
  NUL-terminated current directory into caller-owned storage.
- `execv`: `path argv -- int`
- `execvp`: `file argv -- int`
- `ensure-directory`: `path -- int`
- `wait-child`: `pid -- int`
- `finish-child`: `status --`
- `reset-child-signals`: `--`

Paths and argument arrays use NUL-terminated C strings. `create-file` opens a
write-only, truncated file with mode `0600`. `ensure-directory` accepts an
existing directory that can be opened. `wait-child` waits through interrupted
system calls and returns a normal exit status or `128` plus the terminating
signal. The other
integer-returning operations return a negative value on failure. `finish-child`
flushes standard output and terminates without running parent cleanup code.

## POSIX Descriptors And Sockets

`stdlib/socket.frog` exposes the descriptor and socket operations used by the
HTTP module:

- `descriptor-read`: `fd destination capacity -- count` performs one read,
  retrying interrupted calls. It may return a short count, `0` at EOF, or `-1`
  on failure.
- `descriptor-write`: `fd source length -- count` performs one write, retrying
  interrupted calls. It may return a short count or `-1` on failure.
- `accept-connection`: `listener -- fd` accepts one connection, retrying
  interrupted calls, or returns `-1`.
- `local-socket-pair`: `fds -- status` creates a local stream socket pair in
  storage for two C `int` values.
- `shutdown-write`: `fd -- status` closes the writing side of a socket.
- `ignore-sigpipe`: `-- success` installs the process-wide ignored disposition
  for `SIGPIPE`.

The module is POSIX-specific. Callers own descriptors returned by
`accept-connection` and both descriptors written by `local-socket-pair`.

## Strings And Byte Buffers

`stdlib/string.frog` provides operations for literal `String` values and
heap-backed byte buffers:

- `bytes-copy`: `source destination count --` copies `count` bytes.
- `bytes-equal`: `first first_len second second_len -- bool` compares byte
  ranges.
- `bytes-index-of`: `bytes bytes_len needle needle_len -- int` returns the
  first byte offset of `needle`, or `-1`; an empty needle has offset `0`.
- `bytes-contain`: `bytes bytes_len needle needle_len -- bool` tests whether a
  byte range contains a needle.
- `bytes-starts-with`: `bytes bytes_len prefix prefix_len -- bool` tests whether
  a byte range starts with a prefix; every range starts with an empty prefix.
- `bytes-count`: `bytes bytes_len needle needle_len -- int` counts
  non-overlapping occurrences; an empty needle counts as `0`.
- `string-equal`: `String String -- bool` compares decoded string bytes.
- `string-starts-with`: `String String -- bool` tests a string prefix.
- `byte-buffer-new`: `capacity -- ByteBuffer` creates an empty buffer.
- `byte-buffer-push`: `byte ByteBuffer --` appends one byte and grows the
  buffer when necessary.
- `byte-buffer-append-bytes`: `bytes len ByteBuffer --` appends a byte range.
- `byte-buffer-append-string`: `String ByteBuffer --` appends a string's
  bytes.
- `byte-buffer-equal-string`: `ByteBuffer String -- bool` compares a buffer to
  a string.
- `byte-buffer-free`: `ByteBuffer --` releases a buffer and its storage.

`ByteBuffer` exposes `bytes`, `len`, and `capacity` fields through the normal
record operations. Its byte storage may move after `byte-buffer-push`, so code
must load `@ByteBuffer.bytes` again after an append.

## Opaque-Pointer Containers

`stdlib/containers.frog` provides containers for borrowed `ptr` values. The
containers never release stored values. Callers retain ownership of every value
they insert and of values returned by lookup or removal operations.

- `ptr-array-new`: `capacity -- PtrArray` creates an empty array. A nonpositive
  requested capacity is normalized to a small positive capacity.
- `ptr-array-push`: `value PtrArray --` appends a borrowed pointer, growing the
  array as needed.
- `ptr-array-get`: `PtrArray index -- value found` returns the stored pointer
  and `true` for an in-bounds index, or a null pointer and `false` otherwise.
- `ptr-array-set`: `value PtrArray index -- bool` replaces an in-bounds value
  and returns `true`; it returns `false` for an out-of-bounds index.
- `ptr-array-pop`: `PtrArray -- value found` removes and returns the last value,
  or returns a null pointer and `false` when empty.
- `ptr-array-free`: `PtrArray --` releases the array's internal storage.

Array construction or growth terminates with status 1 when the required slot
count cannot be represented safely as an allocation size.

`PtrArray` exposes `items`, `count`, and `capacity`. `PtrList` exposes `head`
and `count`; its nodes expose `value` and `next`.

- `ptr-list-new`: `-- PtrList` creates an empty list.
- `ptr-list-push-front`: `value PtrList --` adds a borrowed pointer at the
  front.
- `ptr-list-first`: `PtrList -- value found` observes the first pointer, or a
  null pointer and `false` when empty.
- `ptr-list-pop-front`: `PtrList -- value found` removes the first pointer, or
  returns a null pointer and `false` when empty.
- `ptr-list-free`: `PtrList --` releases list nodes but not their values.

`StringMap` maps byte-string keys to borrowed `ptr` values. It copies and owns
each inserted key; callers may change or release the source bytes after
`string-map-set` returns. `StringMap` exposes `buckets`, `count`, and
`capacity`; entries expose `key`, `key_length`, `value`, and `next`.

- `string-map-new`: `capacity -- StringMap` creates an empty map. A nonpositive
  requested capacity is normalized to a small positive bucket count.
- `string-map-get`: `key key_length StringMap -- value found` returns the value
  for a present key, or a null pointer and `false` when absent.
- `string-map-set`: `key key_length value StringMap -- previous replaced`
  inserts or replaces a key. A replacement returns the old borrowed value and
  `true`; a new key returns a null pointer and `false`.
- `string-map-remove`: `key key_length StringMap -- value found` removes a key
  and returns its borrowed value, or a null pointer and `false` when absent.
- `string-map-free`: `StringMap --` releases map entries, owned key copies, and
  buckets, but not values.

Key lengths must be nonnegative. A positive key length requires a pointer to at
least that many readable bytes. Map construction terminates with status 1 when
the required bucket allocation size cannot be represented safely. Maps use a
fixed bucket count with chained entries; they do not resize automatically.

`container-count`: `Container -- int` and `container-empty?`: `Container --
bool` are structural macros that work with `PtrArray`, `PtrList`, and
`StringMap`.

## JSON

`stdlib/json.frog` parses one complete JSON value into an exclusively owned
tree. `JsonValue` is a tagged union with `null`, `boolean`, `number`, `string`,
`array`, and `object` variants. Number payloads preserve their source lexemes;
string payloads contain decoded bytes.

- `json-parse`: `String -- value success` parses a literal string.
- `json-parse-bytes`: `bytes len -- value success` parses a byte range.
- `json-boolean`: `value -- boolean success` reads a boolean payload.
- `json-string-bytes`: `value -- bytes len success` borrows decoded string
  bytes.
- `json-number-bytes`: `value -- bytes len success` borrows the original
  number lexeme.
- `json-number-int`: `value -- integer success` converts an integer-form number
  lexeme when it fits Frog's signed 64-bit `int`. Fraction and exponent forms
  return `false`, even when their mathematical value is integral.
- `json-array-length`: `value -- length success` reads an array length.
- `json-array-get`: `value index -- child found` borrows an indexed child.
- `json-object-get`: `value key -- child found` borrows the last member with a
  literal `String` key.
- `json-object-get-bytes`: `value key key_len -- child found` performs the same
  lookup with a byte-range key.
- `json-free`: `value --` recursively releases a parsed tree.

Parsing accepts RFC 8259 structure, literals, number syntax, JSON whitespace,
escaped Unicode, and valid UTF-16 surrogate pairs in `\u` escapes. Raw
non-control string bytes are preserved without UTF-8 validation. The parser
rejects malformed input, trailing non-whitespace bytes, lone surrogates,
unsupported escapes, and more than 64 nested array/object containers; a root
container counts as one level. Object lookup uses the last member when a key
appears more than once.

On success, the caller owns the returned root and must call `json-free` exactly
once. Array/object children and scalar byte ranges returned by helpers are
borrowed until their root is freed. On failure, parsing frees partial state and
returns a null `JsonValue` handle with `false`. Wrong-variant, missing-key, and
out-of-range helper calls likewise return a neutral value with `false` and do
not change ownership.

## HTTP

`stdlib/http.frog` implements a blocking, one-request HTTP/1.1 server over
caller-supplied POSIX descriptors. It handles an already connected stream or
accepts one connection from an existing listener; address construction,
binding, listening, routing, concurrency, TLS, and timeouts are outside the
module.

```frog
fn HttpHandler HttpRequest* -- HttpResponse* end

proc handler HttpRequest* -- HttpResponse* do
    drop
    200 "frog" http-response
end

proc serve-one-connection int -- int do
    HttpHandler:ref:handler http-serve-connection
end
```

`HttpRequest` exposes `head`, `target`, `target_len`, and `storage`. `head` is
`false` for GET and `true` for HEAD. The target is an undecoded byte slice into
the request storage. A handler borrows the request and its fields only for the
duration of the call; it must not free or retain them. A handler returns an
owned response created by one of the constructors below; allocating an
uninitialized `HttpResponse` directly is unsupported.

- `http-response`: `status String -- HttpResponse` copies a response body.
- `http-response-bytes`: `status bytes len -- HttpResponse` copies a byte-range
  body.
- `http-response-free`: `HttpResponse --` releases a response created by either
  constructor.
- `http-serve-connection`: `fd HttpHandler -- status` owns and closes one
  connected descriptor after serving at most one request.
- `http-serve-one`: `listener HttpHandler -- status` borrows the listener,
  accepts one descriptor, and otherwise behaves like
  `http-serve-connection`.

Response status must be from 200 through 599. Statuses 204, 205, and 304 require
an empty body. Body length must otherwise be nonnegative, and a positive
byte-range body must be readable; violating these constructor preconditions
terminates with status 1. Constructors copy the body, so its source may be
changed or freed immediately afterward.

Serve status is `http-serve-ok`, `http-serve-peer-closed`,
`http-serve-io-error`, or `http-serve-signal-error`. A successfully written
protocol rejection returns `http-serve-ok`.

The accepted request subset is GET or HEAD with exact `HTTP/1.1`, CRLF line
endings, exactly one nonempty Host header, and at most `Content-Length: 0`.
Transfer coding, request bodies, and more than 16 KiB of request headers are
rejected with an empty 400 response. Pipelined requests after the first are not
processed; the connection closes after the first response. Other syntactically
valid headers are ignored. Responses contain an empty reason phrase and
`Connection: close`. They contain `Content-Length` except for statuses 204 and
304; HEAD omits body bytes while retaining the GET body length.

Serving installs the process-wide ignored disposition for `SIGPIPE`. Reads and
writes are blocking, retry interrupted calls, and handle short I/O. The module
does not impose a deadline, so descriptor owners that need slow-client
protection must configure it outside this API.

## Subprocesses

`stdlib/subprocess.frog` runs a program with captured standard output and
standard error:

```frog
from "stdlib/subprocess.frog" import (
    CompletedProcess
    subprocess-argv
    subprocess-arg
    subprocess-arg-pointer
    subprocess-argv-free
    subprocess-run
    subprocess-run-bytes
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
- `subprocess-arg-pointer`: `ptr argv index --` assigns an already
  NUL-terminated C-string pointer as a borrowed argument.
- `subprocess-argv-free`: `ptr --` releases the array. It does not release the
  string literals referenced by the array.
- `subprocess-run`: `argv input -- CompletedProcess` inherits the current
  directory and environment, supplies the `String` as standard input, waits for
  completion, and captures standard output and standard error separately.
- `subprocess-run-in`: `argv input cwd -- CompletedProcess` behaves the same way
  after changing the child to `cwd`. An empty `cwd` inherits the current
  directory.
- `subprocess-run-bytes`: `argv input input_len -- CompletedProcess` supplies a
  raw byte range as standard input, including embedded NUL and non-UTF-8 bytes.
- `subprocess-run-bytes-in`: `argv input input_len cwd -- CompletedProcess`
  combines raw-byte input with a child working directory.
- `completed-process-free`: `CompletedProcess --` releases both captured buffers
  and the result record.

`CompletedProcess.stdout` and `CompletedProcess.stderr` are byte pointers whose
lengths are stored in `stdout_len` and `stderr_len`; the buffers are not
NUL-terminated. `returncode` is the child exit status, or `128` plus the signal
number when the child is terminated by a signal. Commands are executed directly
without a shell, timeout, or environment rewriting.

## Testing

`stdlib/test.frog` provides an explicit `TestSuite` value and four checks:

```frog
from "stdlib/test.frog" import (
    test-suite
    check-int-equal
    check-string-equal
    test-finish
)

proc main -- do
    test-suite
    let suite do
        6 7 * 42 "multiplication" suite check-int-equal
        "frog" "frog" "name" suite check-string-equal
        suite test-finish
    end
end
```

- `check`: `condition name suite --`
- `check-int-equal`: `actual expected name suite --`
- `check-bytes-equal`:
  `actual actual_len expected expected_len name suite --`
- `check-string-equal`: `actual expected name suite --`
- `test-suite`: `-- TestSuite`
- `test-finish`: `TestSuite --`

Each check increments `TestSuite.checks`. A failed check also increments
`TestSuite.failures` and writes `FAIL: <name>\n` to standard error. Checks keep
running after a failure. `test-finish` consumes and releases the suite and exits
with status 1 when any check failed; otherwise it returns normally without
output. Calling `test-finish` more than once or using another alias afterward is
invalid.
