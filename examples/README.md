# Examples

Every example is a complete program with an explicit
`func main -- do ... end`. From the repository root, run one with:

```sh
build/frogc run examples/01_simple.frog
```

- [`01_simple.frog`](./01_simple.frog): stack arithmetic, debug output, and printing.
- [`02_while.frog`](./02_while.frog): a `while` loop, nested conditionals, and arithmetic.
- [`03_fib.frog`](./03_fib.frog): Fibonacci numbers using functions and stack rotation.
- [`04_funcs.frog`](./04_funcs.frog): function composition and a loop.
- [`05_is_prime.frog`](./05_is_prime.frog): primality testing and boolean logic.
- [`06_let.frog`](./06_let.frog): local bindings with `let`.
- [`07_rule30.frog`](./07_rule30.frog): a Rule 30 cellular automaton using byte buffers.
- [`08_gcd_grid.frog`](./08_gcd_grid.frog): Euclidean GCD rendered as a coprimality grid.
- [`09_structs.frog`](./09_structs.frog): struct allocation, field access, and typed functions.
- [`10_tagged_enums.frog`](./10_tagged_enums.frog): enum construction, testing, and projection.
- [`11_c_ffi.frog`](./11_c_ffi.frog): C standard-library calls through explicit interop declarations.
