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
printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\n178\n239\n9\n65\n34\n92\n10\n0\n255\n63\n63\n47\n33\n10\n47\n34\n34\n' \
    | cmp - "$output/strings.out"

test "$(grep -c '^static const uint8_t frog_string_' "$output/strings.c")" -eq 7
test "$(grep -c '= "same";' "$output/strings.c")" -eq 1
test "$(grep -c '= "a";' "$output/strings.c")" -eq 1
grep -q '^static const uint8_t frog_string_2771466528\[\] = ' "$output/strings.c"
grep -q '^static const uint8_t frog_string_2771466528_1\[\] = ' "$output/strings.c"
if grep -q '^.*frog_string_.*\[\] = {' "$output/strings.c"; then
    echo 'string globals must use C string literals' >&2
    exit 1
fi

printf 'proc main -- do "stable" drop drop end\n' \
    | "$compiler" > "$output/stable-before.c"
printf 'proc main -- do "unrelated" drop drop "stable" drop drop end\n' \
    | "$compiler" > "$output/stable-after.c"
before_symbol=$(sed -n 's/^static const uint8_t \(frog_string_[0-9_]*\)\[\] = "stable";$/\1/p' "$output/stable-before.c")
after_symbol=$(sed -n 's/^static const uint8_t \(frog_string_[0-9_]*\)\[\] = "stable";$/\1/p' "$output/stable-after.c")
test -n "$before_symbol"
test "$before_symbol" = "$after_symbol"
