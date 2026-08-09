#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
    echo 'usage: imports.sh /absolute/path/to/frogc' >&2
    exit 2
fi

compiler=$1
fixtures=$(cd "$(dirname "$0")/imports" && pwd)
output=$(cd "$(dirname "$0")/../.." && pwd)/build/bootstrap-imports
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

run_large_path_error() {
    local case_name=large_path
    local path

    printf -v path '%*s' 65536 ''
    path=${path// /a}
    if printf 'from "%s" import value\nproc main -- do value print end\n' "$path" \
        | "$compiler" > "$output/$case_name.c" 2> "$output/$case_name.err"; then
        echo "expected $case_name to fail" >&2
        exit 1
    fi
    printf 'frogc: import path exceeds max-import-path-bytes\n' | cmp - "$output/$case_name.err"
}

run_ok direct_proc $'42\n'
run_ok group_alias_identity $'6\n4\n6\n'
run_ok nested_reexport_scope $'7\n'
run_ok macro_scope $'18\n'
run_ok root_relative_nested $'11\n'
run_ok macro_reexport_scope $'14\n'
run_ok extern_reexport $'9\n'
run_ok one_byte_path $'21\n'
run_ok string_module_ids $'PDR\n'
run_ok path_length_limit $'33\n'
run_ok ignored_control_flow $'44\n'
run_ok utf8_path $'55\n'
run_ok records_alias $'8\n'
run_ok records_reexport $'13\n'
run_ok unions_alias $'true\n8\n'
run_ok unions_reexport $'13\n'
run_ok unions_macro_private $'21\n'

run_error missing_file 'import file not found'
run_error missing_name 'imported name not found'
run_error extern_contract_conflict 'incompatible declarations for C symbol'
run_error alias_conflict 'import alias conflict'
run_error local_alias_conflict 'import alias conflict'
run_error direct_cycle 'cyclic import'
run_error self_cycle 'cyclic import'
run_error wildcard 'wildcard imports are not supported'
run_error comma 'commas are not valid in import lists'
run_error module_alias 'module aliases are not supported'
run_error nested_import 'imports are only allowed at top level'
run_error unterminated_group 'expected ) after import list'
run_error ignored_if_import 'imports are only allowed at top level'
run_error ignored_while_import 'imports are only allowed at top level'
run_error ignored_let_import 'imports are only allowed at top level'
run_error reserved_alias 'invalid imported name'
run_error ungrouped_parenthesis 'invalid imported name'
run_error invalid_utf8_path 'import path must be valid UTF-8'
run_error truncated_utf8_path 'import path must be valid UTF-8'
run_error overlong_utf8_path 'import path must be valid UTF-8'
run_error surrogate_utf8_path 'import path must be valid UTF-8'
run_error above_unicode_utf8_path 'import path must be valid UTF-8'
run_error path_too_long 'import path exceeds max-import-path-bytes'
run_error records_collision 'import alias conflict'
run_error records_alias_reserved 'invalid imported name'
run_error records_alias_invalid_identifier 'record import alias must be an identifier'
run_error unions_collision 'import alias conflict'
run_error unions_alias_reserved 'invalid imported name'
run_error unions_alias_invalid_identifier 'union import alias must be an identifier'
run_error unions_nominal_mismatch 'compile-time stack type mismatch'
run_large_path_error
