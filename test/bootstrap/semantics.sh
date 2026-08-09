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
    shift 2
    local fixture="$fixtures/$case_name"

    (
        cd "$fixture"
        "$compiler" < main.frog > "$output/$case_name.c"
    )
    local sources=("$output/$case_name.c")
    if [[ -f "$fixture/helper.c" ]]; then
        sources+=("$fixture/helper.c")
    fi
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 \
        "${sources[@]}" -o "$output/$case_name"
    "$output/$case_name" "$@" > "$output/$case_name.out"
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
run_ok integer_bases $'0\n7\n146\n819\n43981\n9223372036854775807\n9223372036854775807\n9223372036854775807\n'
run_ok prelude_shadow $'42\n7\n2\n1\n6\n20\n10\n20\n10\n40\n50\n2\n1\n4\n3\n5\n7\n6\n16\n'
run_ok macro_shadow $'11\n21\n31\n'
run_ok characters $'65\n233\n8364\n128512\n'
run_ok args $'3\n/\nfrog\npond\n' frog pond
run_ok pointer_store $'65\n'
run_ok c_ffi $'42\n42\n711\ntrue\nfalse\n'
run_ok records_layout $'41\ntrue\ntrue\n32\n7\n'
printf '%s' $'void p1(void) {\n  Cell frog_ffi_arg_2 = frog_pop();\n  Cell frog_ffi_arg_1 = frog_pop();\n  Cell frog_ffi_arg_0 = frog_pop();\n  frog_push((Cell)ffi_test_mix((int)frog_ffi_arg_0, (int)(frog_ffi_arg_1 != 0), (void *)(intptr_t)frog_ffi_arg_2));\n}\n' \
    | cmp - <(sed -n '/^void p1(void) {$/,/^}$/p' "$output/c_ffi.c")

run_error integer_overflow 'integer literal exceeds the signed 64-bit range'
run_source_error integer_binary_missing_digits \
    $'proc main -- do 0b print end\n' \
    'invalid integer literal'
run_source_error integer_binary_invalid_digit \
    $'proc main -- do 0b102 print end\n' \
    'invalid integer literal'
run_source_error integer_octal_invalid_digit \
    $'proc main -- do 0o8 print end\n' \
    'invalid integer literal'
run_source_error integer_hex_missing_digits \
    $'proc main -- do 0x print end\n' \
    'invalid integer literal'
run_source_error integer_hex_invalid_digit \
    $'proc main -- do 0xg print end\n' \
    'invalid integer literal'
run_source_error integer_hex_trailing_punctuation \
    $'proc main -- do 0x1.2 print end\n' \
    'invalid integer literal'
run_source_error integer_binary_overflow \
    $'proc main -- do 0b1000000000000000000000000000000000000000000000000000000000000000 print end\n' \
    'integer literal exceeds the signed 64-bit range'
run_source_error integer_octal_overflow \
    $'proc main -- do 0o1000000000000000000000 print end\n' \
    'integer literal exceeds the signed 64-bit range'
run_source_error integer_hex_overflow \
    $'proc main -- do 0x8000000000000000 print end\n' \
    'integer literal exceeds the signed 64-bit range'
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
run_source_error extern_invalid_c_symbol \
    $'extern magnitude not-valid c-int -- c-int end\nproc main -- do end\n' \
    'invalid C symbol'
run_source_error extern_c_keyword \
    $'extern invalid int -- c-int end\nproc main -- do end\n' \
    'invalid C symbol'
run_source_error extern_internal_symbol \
    $'extern invalid frog_push c-int -- end\nproc main -- do end\n' \
    'invalid C symbol'
run_source_error extern_generated_proc_symbol \
    $'extern invalid p0 -- c-int end\nproc main -- do end\n' \
    'invalid C symbol'
run_source_error extern_unknown_abi_type \
    $'extern magnitude abs int -- c-int end\nproc main -- do end\n' \
    'unknown C ABI type'
run_source_error extern_multiple_outputs \
    $'extern pair abs c-int -- c-int c-int end\nproc main -- do end\n' \
    'external procedure may return at most one value'
run_source_error extern_missing_end \
    $'extern magnitude abs c-int -- c-int\n' \
    'expected end after external signature'
run_source_error extern_main \
    $'extern main abs c-int -- c-int end\n' \
    'main cannot be external'
run_source_error extern_inside_proc \
    $'proc main -- do extern magnitude abs c-int -- c-int end end\n' \
    'declarations are only allowed at top level'
run_source_error extern_inside_macro \
    $'macro bad extern magnitude abs c-int -- c-int end end\nproc main -- do end\n' \
    'declarations are not allowed in macro bodies'
run_source_error extern_contract_type \
    $'extern magnitude abs c-int -- c-int end\nproc main -- do true magnitude drop end\n' \
    'compile-time stack type mismatch'
run_source_error extern_incompatible_contract \
    $'extern as-int abs c-int -- c-int end\nextern as-ptr abs c-int -- c-ptr end\nproc main -- do end\n' \
    'incompatible declarations for C symbol'
run_source_error record_wrong_owner \
    $'record Point x int end\nrecord Box x int end\nproc main -- do Box:alloc Point.x drop end\n' \
    'compile-time stack type mismatch'
run_source_error record_wrong_value \
    $'record Point x int end\nproc main -- do Point:alloc let point do true point Point.x! end end\n' \
    'compile-time stack type mismatch'
run_source_error record_unknown_field \
    $'record Point x int end\nproc main -- do Point:alloc Point.y drop end\n' \
    'unknown record field'
run_source_error record_unknown_type \
    $'record Point x Missing end\nproc main -- do end\n' \
    'unknown type in record field'
run_source_error record_duplicate_field \
    $'record Point x int x bool end\nproc main -- do end\n' \
    'duplicate record field: x'
run_source_error record_unsupported_cast \
    $'record Point x int end\nproc main -- do 1 Point cast drop end\n' \
    'unsupported cast'
