# Examples

Every example is a complete program with an explicit `proc main -- do ... end`. Run one with `build/frogc run examples/01_simple.frog`; the CLI compiles it to C and executes the compiled binary.

- `01_simple.frog`: Basic stack arithmetic, debug, and print demo.
- `02_while.frog`: While loop, nested if/else, and arithmetic demo.
- `03_fib.frog`: Fibonacci sequence using procedures and stack rotation.
- `04_procs.frog`: Small procedure composition and loop demo.
- `05_is_prime.frog`: Prime-checking procedures and boolean logic demo.
- `06_let.frog`: Local binding demo with `let`.
- `07_rule30.frog`: Rule 30 ASCII cellular automaton using byte buffers.
- `08_gcd_grid.frog`: Euclidean GCD rendered as a coprimality grid.
- `09_records.frog`: Record allocation, field access, and a typed procedure.
- `10_tagged_unions.frog`: A tagged result with checked testing and projection.
- `11_c_ffi.frog`: Calls C standard-library functions through scalar C FFI.
