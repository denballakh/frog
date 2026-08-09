#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
    echo 'usage: semantics.sh /absolute/path/to/frogc' >&2
    exit 2
fi

compiler=$1
fixtures=$(cd "$(dirname "$0")/semantics" && pwd)
output=$(cd "$(dirname "$0")/../.." && pwd)/build/bootstrap-semantics
mkdir -p "$output"

run_ok() {
    local case_name=$1
    local expected=$2
    local fixture="$fixtures/$case_name"

    (
        cd "$fixture"
        "$compiler" < main.frog > "$output/$case_name.c"
    )
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 \
        "$output/$case_name.c" -o "$output/$case_name"
    "$output/$case_name" > "$output/$case_name.out"
    printf '%s' "$expected" | cmp - "$output/$case_name.out"
}

run_error() {
    local case_name=$1
    local expected=$2
    local fixture="$fixtures/$case_name"

    if (
        cd "$fixture"
        "$compiler" < main.frog > "$output/$case_name.c" 2> "$output/$case_name.err"
    ); then
        echo "expected $case_name to fail" >&2
        exit 1
    fi
    printf 'frogc: %s\n' "$expected" | cmp - "$output/$case_name.err"
}

run_source_error() {
    local case_name=$1
    local source=$2
    local expected=$3

    if printf '%s' "$source" \
        | "$compiler" > "$output/$case_name.c" 2> "$output/$case_name.err"; then
        echo "expected $case_name to fail" >&2
        exit 1
    fi
    printf 'frogc: %s\n' "$expected" | cmp - "$output/$case_name.err"
}

run_ok bool_cast $'1\n0\n1\n'
run_ok integer_max $'9223372036854775807\n'
run_ok macro_shadow $'11\n21\n31\n'
run_ok characters $'65\n233\n8364\n128512\n'

run_error integer_overflow 'integer literal exceeds the signed 64-bit range'
run_error character_empty 'invalid character literal'
run_error character_two_codepoints 'invalid character literal'
run_source_error character_malformed_utf8 \
    $'proc main -- do \'\x80\' drop end\n' \
    'invalid character literal'
run_source_error character_truncated_utf8 \
    $'proc main -- do \'\xE2\x82\' drop end\n' \
    'invalid character literal'
run_source_error if_without_do \
    $'proc main -- do if true end end\n' \
    'if requires do before end'
run_source_error else_without_do \
    $'proc main -- do if true else end end\n' \
    'else requires a preceding if arm and do'
run_source_error while_without_do \
    $'proc main -- do while end end\n' \
    'while requires do before end'
