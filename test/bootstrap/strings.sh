#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
    echo 'usage: strings.sh /absolute/path/to/frogc' >&2
    exit 2
fi

compiler=$1
fixtures=$(cd "$(dirname "$0")/strings" && pwd)
output=$(cd "$(dirname "$0")/../.." && pwd)/build/bootstrap-strings
mkdir -p "$output"

(
    cd "$fixtures"
    "$compiler" < main.frog > "$output/strings.c"
)
gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 \
    "$output/strings.c" -o "$output/strings"
"$output/strings" > "$output/strings.out"
printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\n98\ntrue\ntrue\n178\n239\n9\n65\n34\n92\n10\n0\n255\n63\n63\n47\n33\n10\n47\n34\n34\n' \
    | cmp - "$output/strings.out"

grep -q '^typedef struct {' "$output/strings.c"
grep -q '^} FrogString;$' "$output/strings.c"
grep -q '^  uint8_t \*bytes;$' "$output/strings.c"
grep -q '^  Cell len;$' "$output/strings.c"
test "$(grep -c '^static const FrogString frog_string_' "$output/strings.c")" -eq 7
test "$(grep -c '^static uint8_t frog_string_' "$output/strings.c")" -eq 7
test "$(grep -c '= "same";' "$output/strings.c")" -eq 1
grep -q '^static uint8_t frog_string_2771466528_bytes\[\] = ' "$output/strings.c"
grep -q '^static const FrogString frog_string_2771466528 = { frog_string_2771466528_bytes, 8 };$' "$output/strings.c"
grep -q '^static uint8_t frog_string_2771466528_1_bytes\[\] = ' "$output/strings.c"
grep -q '^static const FrogString frog_string_2771466528_1 = { frog_string_2771466528_1_bytes, 8 };$' "$output/strings.c"
same_symbol=$(sed -n 's/^static uint8_t \(frog_string_[0-9_]*\)_bytes\[\] = "same";$/\1/p' "$output/strings.c")
test -n "$same_symbol"
test "$(grep -F -c "= (Cell)(intptr_t)&$same_symbol;" "$output/strings.c")" -eq 3
if [[ $(grep -c 'frog_alloc(' "$output/strings.c") -ne 1 ]]; then
    echo 'string literal use must not allocate' >&2
    exit 1
fi
if grep -Eq 'FrogStack|frog_stack|frog_push|frog_pop' "$output/strings.c"; then
    echo 'generated runtime stack symbol in strings.c' >&2
    exit 1
fi

expect_compile_error() {
    local name=$1
    local expected=$2
    local source=$3

    if (
        cd "$fixtures"
        printf '%s\n' "$source" | "$compiler" > "$output/$name.c" 2> "$output/$name.err"
    ); then
        echo "expected $name to fail" >&2
        exit 1
    fi
    printf 'frogc: %s\n' "$expected" | cmp - "$output/$name.err"
}

expect_compile_error reserved-string-record 'invalid record name' \
    $'record String value int end\nproc main -- do end'
expect_compile_error reserved-string-union 'invalid union name' \
    $'union String case value end\nproc main -- do end'
expect_compile_error reserved-string-function 'invalid function name' \
    $'fn String -- end\nproc main -- do end'
expect_compile_error reserved-string-procedure 'reserved keyword cannot be a procedure name' \
    'proc String -- do end'
expect_compile_error reserved-string-macro 'reserved keyword cannot be a macro name' \
    $'macro String 1 end\nproc main -- do end'
expect_compile_error reserved-string-local 'String cannot be a local name' \
    'proc main -- do 1 let String do end end'
expect_compile_error reserved-string-import 'invalid imported name' \
    $'from "lib.frog" import shared as String\nproc main -- do end'
expect_compile_error reserved-c-string-symbol 'invalid C symbol' \
    $'extern bad FrogString -- c-int end\nproc main -- do bad drop end'

if printf 'proc main -- do "x" @u8 drop end\n' \
    | "$compiler" > "$output/string-as-ptr.c" 2> "$output/string-as-ptr.err"; then
    echo 'expected String used as ptr to fail' >&2
    exit 1
fi
printf 'frogc: compile-time stack type mismatch\n' | cmp - "$output/string-as-ptr.err"

if printf 'proc main -- do "x" 1 + drop end\n' \
    | "$compiler" > "$output/string-as-int.c" 2> "$output/string-as-int.err"; then
    echo 'expected String used as int to fail' >&2
    exit 1
fi
printf 'frogc: invalid operand types for pointer/integer arithmetic\n' \
    | cmp - "$output/string-as-int.err"

printf 'proc main -- do "stable" drop end\n' \
    | "$compiler" > "$output/stable-before.c"
printf 'proc main -- do "unrelated" drop "stable" drop end\n' \
    | "$compiler" > "$output/stable-after.c"
before_symbol=$(sed -n 's/^static uint8_t \(frog_string_[0-9_]*\)_bytes\[\] = "stable";$/\1/p' "$output/stable-before.c")
after_symbol=$(sed -n 's/^static uint8_t \(frog_string_[0-9_]*\)_bytes\[\] = "stable";$/\1/p' "$output/stable-after.c")
test -n "$before_symbol"
test "$before_symbol" = "$after_symbol"

(
    cd "$fixtures"
    "$compiler" < types.frog > "$output/string-types.c"
)
gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 \
    "$output/string-types.c" -o "$output/string-types"
"$output/string-types" > "$output/string-types.out"
printf '5\n5\n5\n4\n' | cmp - "$output/string-types.out"
