from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

COMMAND_TIMEOUT_SECONDS = 30
STRICT_C_FLAGS = (
    '-std=c11',
    '-pedantic',
    '-Wall',
    '-Wextra',
    '-Wconversion',
    '-Werror',
    '-O2',
)


@dataclass(frozen=True)
class RunCase:
    group: str
    name: str
    source: str
    expected_stdout: bytes | None
    args: tuple[str, ...] = ()
    expected_status: int = 0


@dataclass(frozen=True)
class FixtureErrorCase:
    group: str
    name: str
    expected: str
    source: str = 'main.frog'


@dataclass(frozen=True)
class SourceErrorCase:
    name: str
    source: bytes
    expected: str
    cwd: str = 'semantics'


IMPORT_RUN_CASES = [
    RunCase('imports', 'direct_proc', 'imports/direct_proc/main.frog', b'42\n'),
    RunCase('imports', 'group_alias_identity', 'imports/group_alias_identity/main.frog', b'6\n4\n6\n'),
    RunCase('imports', 'nested_reexport_scope', 'imports/nested_reexport_scope/main.frog', b'7\n'),
    RunCase('imports', 'macro_scope', 'imports/macro_scope/main.frog', b'18\n'),
    RunCase('imports', 'root_relative_nested', 'imports/root_relative_nested/main.frog', b'11\n'),
    RunCase('imports', 'macro_reexport_scope', 'imports/macro_reexport_scope/main.frog', b'14\n'),
    RunCase('imports', 'extern_reexport', 'imports/extern_reexport/main.frog', b'9\n'),
    RunCase('imports', 'one_byte_path', 'imports/one_byte_path/main.frog', b'21\n'),
    RunCase('imports', 'string_module_ids', 'imports/string_module_ids/main.frog', b'PDR\n'),
    RunCase('imports', 'path_length_limit', 'imports/path_length_limit/main.frog', b'33\n'),
    RunCase('imports', 'ignored_control_flow', 'imports/ignored_control_flow/main.frog', b'44\n'),
    RunCase('imports', 'utf8_path', 'imports/utf8_path/main.frog', b'55\n'),
    RunCase('imports', 'records_alias', 'imports/records_alias/main.frog', b'8\n'),
    RunCase('imports', 'records_reexport', 'imports/records_reexport/main.frog', b'13\n'),
    RunCase('imports', 'unions_alias', 'imports/unions_alias/main.frog', b'true\n8\n'),
    RunCase('imports', 'unions_reexport', 'imports/unions_reexport/main.frog', b'13\n'),
    RunCase('imports', 'unions_macro_private', 'imports/unions_macro_private/main.frog', b'21\n'),
    RunCase('imports', 'functions_direct', 'imports/functions_direct/main.frog', b'42\n'),
    RunCase('imports', 'functions_alias', 'imports/functions_alias/main.frog', b'6\n'),
    RunCase('imports', 'functions_reexport', 'imports/functions_reexport/main.frog', b'43\n'),
    RunCase('imports', 'functions_macro_private', 'imports/functions_macro_private/main.frog', b'42\n'),
    RunCase('imports', 'functions_extern', 'imports/functions_extern/main.frog', b'77\n'),
    RunCase('imports', 'optimizer_macro_scope', 'imports/optimizer_macro_scope/main.frog', b'41\n'),
    RunCase('imports', 'constants_reexport', 'imports/constants_reexport/main.frog', b'42\n'),
    RunCase('imports', 'peek_macro', 'imports/peek_macro/main.frog', b'7\n4\n3\n'),
]


IMPORT_ERROR_CASES = [
    FixtureErrorCase('imports', 'missing_file', 'import file not found'),
    FixtureErrorCase('imports', 'missing_name', 'imported name not found'),
    FixtureErrorCase('imports', 'extern_contract_conflict', 'incompatible declarations for C symbol'),
    FixtureErrorCase('imports', 'alias_conflict', 'import alias conflict'),
    FixtureErrorCase('imports', 'local_alias_conflict', 'import alias conflict'),
    FixtureErrorCase('imports', 'direct_cycle', 'cyclic import'),
    FixtureErrorCase('imports', 'self_cycle', 'cyclic import'),
    FixtureErrorCase('imports', 'wildcard', 'wildcard imports are not supported'),
    FixtureErrorCase('imports', 'comma', 'commas are not valid in import lists'),
    FixtureErrorCase('imports', 'module_alias', 'module aliases are not supported'),
    FixtureErrorCase('imports', 'nested_import', 'imports are only allowed at top level'),
    FixtureErrorCase('imports', 'unterminated_group', 'expected ) after import list'),
    FixtureErrorCase('imports', 'ignored_if_import', 'imports are only allowed at top level'),
    FixtureErrorCase('imports', 'ignored_while_import', 'imports are only allowed at top level'),
    FixtureErrorCase('imports', 'ignored_let_import', 'imports are only allowed at top level'),
    FixtureErrorCase('imports', 'reserved_alias', 'invalid imported name'),
    FixtureErrorCase('imports', 'ungrouped_parenthesis', 'invalid imported name'),
    FixtureErrorCase('imports', 'invalid_utf8_path', 'import path must be valid UTF-8'),
    FixtureErrorCase('imports', 'truncated_utf8_path', 'import path must be valid UTF-8'),
    FixtureErrorCase('imports', 'overlong_utf8_path', 'import path must be valid UTF-8'),
    FixtureErrorCase('imports', 'surrogate_utf8_path', 'import path must be valid UTF-8'),
    FixtureErrorCase('imports', 'above_unicode_utf8_path', 'import path must be valid UTF-8'),
    FixtureErrorCase('imports', 'path_too_long', 'import path exceeds max-import-path-bytes'),
    FixtureErrorCase('imports', 'records_collision', 'import alias conflict'),
    FixtureErrorCase('imports', 'records_alias_reserved', 'invalid imported name'),
    FixtureErrorCase('imports', 'records_alias_invalid_identifier', 'record import alias must be an identifier'),
    FixtureErrorCase('imports', 'unions_collision', 'import alias conflict'),
    FixtureErrorCase('imports', 'unions_alias_reserved', 'invalid imported name'),
    FixtureErrorCase('imports', 'unions_alias_invalid_identifier', 'union import alias must be an identifier'),
    FixtureErrorCase('imports', 'unions_nominal_mismatch', 'compile-time stack type mismatch'),
    FixtureErrorCase('imports', 'functions_alias_reserved', 'invalid imported name'),
    FixtureErrorCase('imports', 'functions_alias_invalid_identifier', 'function import alias must be an identifier'),
    FixtureErrorCase('imports', 'functions_alias_collision', 'import alias conflict'),
    FixtureErrorCase('imports', 'functions_nominal_mismatch', 'compile-time stack type mismatch'),
]


SEMANTIC_RUN_CASES = [
    RunCase('semantics', 'bool_cast', 'semantics/bool_cast/main.frog', b'1\n0\n1\n'),
    RunCase('semantics', 'integer_max', 'semantics/integer_max/main.frog', b'9223372036854775807\n'),
    RunCase(
        'semantics',
        'integer_bases',
        'semantics/integer_bases/main.frog',
        b'0\n7\n146\n819\n43981\n9223372036854775807\n9223372036854775807\n9223372036854775807\n',
    ),
    RunCase(
        'semantics',
        'prelude_shadow',
        'semantics/prelude_shadow/main.frog',
        b'42\n7\n2\n1\n6\n20\n10\n20\n10\n40\n50\n2\n1\n4\n3\n5\n7\n6\n16\n',
    ),
    RunCase('semantics', 'macro_shadow', 'semantics/macro_shadow/main.frog', b'11\n21\n31\n'),
    RunCase('semantics', 'characters', 'semantics/characters/main.frog', b'65\n233\n8364\n128512\n'),
    RunCase('semantics', 'args', 'semantics/args/main.frog', b'3\n/\nfrog\npond\n', ('frog', 'pond')),
    RunCase('semantics', 'pointer_store', 'semantics/pointer_store/main.frog', b'65\n'),
    RunCase('semantics', 'c_ffi', 'semantics/c_ffi/main.frog', b'42\n42\n9\n711\ntrue\nfalse\n'),
    RunCase('semantics', 'records_layout', 'semantics/records_layout/main.frog', b'41\ntrue\ntrue\n32\n7\n'),
    RunCase(
        'semantics',
        'unions_layout',
        'semantics/unions_layout/main.frog',
        b'true\nfalse\ntrue\n9\ntrue\n7\ntrue\ntrue\ntrue\n',
    ),
    RunCase(
        'semantics',
        'functions_layout',
        'semantics/functions_layout/main.frog',
        b'42\n2\n3\n42\n7\n7\n77\n8\n5\n42\n42\n5\n6\ntrue\ntrue\n0\n11\n22\n99\n11\n22\n77\n17\n',
    ),
    RunCase('semantics', 'functions_macro_shadow', 'semantics/functions_macro_shadow/main.frog', b'99\n'),
    RunCase('semantics', 'optimizer_constant_add', 'semantics/optimizer_constant_add/main.frog', b'5\n18\n3\n9\n'),
    RunCase('semantics', 'optimizer_macro_shadow', 'semantics/optimizer_macro_shadow/main.frog', b'41\n'),
    RunCase('semantics', 'type_stack_growth', 'semantics/type_stack_growth/main.frog', b'42\n'),
    RunCase('semantics', 'procedure_symbols', 'semantics/procedure_symbols/main.frog', b''),
    RunCase(
        'semantics',
        'constants',
        'semantics/constants/main.frog',
        b'42\n42\n42\n42\ntrue\n42\n4\n-5\ntrue\n9223372036854775807\n99\n9\n8\n7\n6\n5\n4\n3\n2\n1\n0\n3\n9\n',
    ),
    RunCase(
        'semantics',
        'constants_operators',
        'semantics/constants_operators/main.frog',
        b'13\n5\n36\n2\n1\n1\n2\n8\n2\n7\n2\n5\n-1\nfalse\ntrue\ntrue\nfalse\ntrue\ntrue\nfalse\ntrue\nfalse\n',
    ),
    RunCase('semantics', 'peek', 'semantics/peek/main.frog', b'3\n3\ntrue\ntrue\n7\n4\n3\n2\n1\n2\n1\n'),
]


STATUS_CASES = [
    RunCase('semantics', 'unions_wrong_variant', 'semantics/unions_wrong_variant/main.frog', None, expected_status=1),
    RunCase('semantics', 'unions_invalid_tag', 'semantics/unions_invalid_tag/main.frog', None, expected_status=1),
    RunCase('semantics', 'unions_negative_tag', 'semantics/unions_negative_tag/main.frog', None, expected_status=1),
    RunCase('semantics', 'unions_null', 'semantics/unions_null/main.frog', None, expected_status=1),
    RunCase(
        'semantics', 'functions_unknown_proc', 'semantics/functions_unknown_proc/main.frog', None, expected_status=1
    ),
    RunCase(
        'semantics',
        'functions_incompatible_proc',
        'semantics/functions_incompatible_proc/main.frog',
        None,
        expected_status=1,
    ),
]


FIXTURE_ERROR_CASES = [
    *IMPORT_ERROR_CASES,
    FixtureErrorCase('semantics', 'integer_overflow', 'integer literal exceeds the signed 64-bit range'),
    FixtureErrorCase('semantics', 'character_empty', 'invalid character literal'),
    FixtureErrorCase('semantics', 'character_two_codepoints', 'invalid character literal'),
    FixtureErrorCase('standalone', 'macro_recursive', 'recursive macro expansion', 'macro_recursive.frog'),
    FixtureErrorCase('standalone', 'macro_invalid', 'else outside macro if block', 'macro_invalid.frog'),
    FixtureErrorCase(
        'standalone',
        'macro_reserved_name',
        'reserved keyword cannot be a macro name',
        'macro_reserved_name.frog',
    ),
]


STRING_RUN_CASES = [
    RunCase(
        'strings',
        'strings',
        'strings/main.frog',
        b'true\ntrue\ntrue\ntrue\ntrue\ntrue\n98\ntrue\ntrue\n178\n239\n9\n65\n34\n92\n10\n0\n255\n63\n63\n47\n33\n10\n47\n34\n34\n',
    ),
    RunCase('strings', 'string_types', 'strings/types.frog', b'5\n5\n5\n4\n'),
]


STANDALONE_RUN_CASES = [
    RunCase('standalone', 'read_file', 'standalone/read_file.frog', b'#\ntrue\n0\nfalse\n'),
    RunCase('standalone', 'macros', 'standalone/macros.frog', b'1\n2\n3\n3\n2\n1\n1\n2\n1\n6\n7\n9\n7\n'),
]


SOURCE_ERROR_CASES = [
    SourceErrorCase(
        'constant_divide_by_zero',
        b'const bad 1 0 / end\nproc main -- do end\n',
        'constant division by zero',
    ),
    SourceErrorCase(
        'constant_overflow',
        b'const bad 9223372036854775807 1 + end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_subtract_overflow',
        b'const bad 0 9223372036854775807 - 1 - 1 - end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_multiply_positive_overflow',
        b'const bad 9223372036854775807 2 * end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_multiply_positive_negative_overflow',
        b'const bad 9223372036854775807 0 2 - * end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_multiply_negative_positive_overflow',
        b'const bad 0 9223372036854775807 - 2 * end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_multiply_negative_overflow',
        b'const bad 0 9223372036854775807 - 0 2 - * end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_multiply_min_overflow',
        b'const bad 0 9223372036854775807 - 1 - 0 1 - * end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_divide_overflow',
        b'const bad 0 9223372036854775807 - 1 - 0 1 - / end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_modulo_overflow',
        b'const bad 0 9223372036854775807 - 1 - 0 1 - % end\nproc main -- do end\n',
        'constant integer overflow',
    ),
    SourceErrorCase(
        'constant_invalid_shift',
        b'const bad 1 63 << end\nproc main -- do end\n',
        'invalid shift in constant expression',
    ),
    SourceErrorCase(
        'constant_negative_shift_value',
        b'const bad 0 1 - 1 << end\nproc main -- do end\n',
        'invalid shift in constant expression',
    ),
    SourceErrorCase(
        'constant_negative_shift_count',
        b'const bad 1 0 1 - << end\nproc main -- do end\n',
        'invalid shift in constant expression',
    ),
    SourceErrorCase(
        'constant_shift_overflow', b'const bad 2 62 << end\nproc main -- do end\n', 'constant integer overflow'
    ),
    SourceErrorCase(
        'constant_stack_underflow', b'const bad 1 + end\nproc main -- do end\n', 'constant expression stack underflow'
    ),
    SourceErrorCase(
        'constant_type_mismatch',
        b'const bad true 1 + end\nproc main -- do end\n',
        'constant expression type mismatch',
    ),
    SourceErrorCase(
        'constant_empty_result', b'const bad end\nproc main -- do end\n', 'constant must produce at least one value'
    ),
    SourceErrorCase(
        'constant_recursion',
        b'const first second end\nconst second first end\nproc main -- do end\n',
        'recursive constant',
    ),
    SourceErrorCase(
        'constant_unsupported_word',
        b'const bad print end\nproc main -- do end\n',
        'unsupported constant expression word',
    ),
    SourceErrorCase(
        'constant_builtin_precedence',
        b'const print 41 end\nconst value print end\nproc main -- do value print end\n',
        'unsupported constant expression word',
    ),
    SourceErrorCase(
        'constant_record_getter_precedence',
        b'record Point x int end\nconst @Point.x 41 end\nconst value @Point.x end\nproc main -- do end\n',
        'unsupported constant expression word',
    ),
    SourceErrorCase(
        'constant_unsupported_macro',
        b'macro twice dup + end\nconst bad 1 twice end\nproc main -- do end\n',
        'unsupported constant expression word',
    ),
    SourceErrorCase(
        'constant_duplicate',
        b'const value 1 end\nconst value 2 end\nproc main -- do end\n',
        'duplicate constant name',
    ),
    SourceErrorCase(
        'constant_procedure_conflict',
        b'const value 1 end\nproc value -- int do 2 end\nproc main -- do end\n',
        'duplicate declaration name: value',
    ),
    SourceErrorCase(
        'constant_inside_proc',
        b'proc main -- do const bad 1 end end\n',
        'declarations are only allowed at top level',
    ),
    SourceErrorCase(
        'constant_inside_macro',
        b'macro bad const value 1 end end\nproc main -- do end\n',
        'declarations are not allowed in macro bodies',
    ),
    SourceErrorCase('peek_empty', b'proc main -- do peek do end end\n', 'peek requires at least one name'),
    SourceErrorCase(
        'peek_underflow', b'proc main -- do 1 peek first second do end end\n', 'compile-time stack underflow'
    ),
    SourceErrorCase('peek_unterminated', b'proc main -- do 1 peek value end end\n', 'unterminated peek binding'),
    SourceErrorCase('peek_scope', b'proc main -- do 1 peek value do end value drop end\n', 'unknown word'),
    SourceErrorCase(
        'peek_in_constant',
        b'const bad 1 peek value do end end\nproc main -- do end\n',
        'unsupported constant expression word',
    ),
    SourceErrorCase('integer_binary_missing_digits', b'proc main -- do 0b print end\n', 'invalid integer literal'),
    SourceErrorCase('integer_binary_invalid_digit', b'proc main -- do 0b102 print end\n', 'invalid integer literal'),
    SourceErrorCase('integer_octal_invalid_digit', b'proc main -- do 0o8 print end\n', 'invalid integer literal'),
    SourceErrorCase('integer_hex_missing_digits', b'proc main -- do 0x print end\n', 'invalid integer literal'),
    SourceErrorCase('integer_hex_invalid_digit', b'proc main -- do 0xg print end\n', 'invalid integer literal'),
    SourceErrorCase(
        'integer_hex_trailing_punctuation', b'proc main -- do 0x1.2 print end\n', 'invalid integer literal'
    ),
    SourceErrorCase(
        'integer_binary_overflow',
        b'proc main -- do 0b1000000000000000000000000000000000000000000000000000000000000000 print end\n',
        'integer literal exceeds the signed 64-bit range',
    ),
    SourceErrorCase(
        'integer_octal_overflow',
        b'proc main -- do 0o1000000000000000000000 print end\n',
        'integer literal exceeds the signed 64-bit range',
    ),
    SourceErrorCase(
        'integer_hex_overflow',
        b'proc main -- do 0x8000000000000000 print end\n',
        'integer literal exceeds the signed 64-bit range',
    ),
    SourceErrorCase('character_malformed_utf8', b"proc main -- do '\x80' drop end\n", 'invalid character literal'),
    SourceErrorCase('character_truncated_utf8', b"proc main -- do '\xe2\x82' drop end\n", 'invalid character literal'),
    SourceErrorCase('if_without_do', b'proc main -- do if true end end\n', 'if requires do before end'),
    SourceErrorCase(
        'else_without_do',
        b'proc main -- do if true else end end\n',
        'else requires a preceding if arm and do',
    ),
    SourceErrorCase('while_without_do', b'proc main -- do while end end\n', 'while requires do before end'),
    SourceErrorCase(
        'extern_invalid_c_symbol',
        b'extern magnitude not-valid c-int -- c-int end\nproc main -- do end\n',
        'invalid C symbol',
    ),
    SourceErrorCase('extern_c_keyword', b'extern invalid int -- c-int end\nproc main -- do end\n', 'invalid C symbol'),
    SourceErrorCase(
        'extern_internal_symbol', b'extern invalid frog_push c-int -- end\nproc main -- do end\n', 'invalid C symbol'
    ),
    SourceErrorCase(
        'extern_generated_proc_symbol',
        b'extern invalid frog_proc_0_main -- c-int end\nproc main -- do end\n',
        'invalid C symbol',
    ),
    SourceErrorCase(
        'extern_unknown_abi_type',
        b'extern magnitude abs int -- c-int end\nproc main -- do end\n',
        'unknown C ABI type',
    ),
    SourceErrorCase(
        'extern_multiple_outputs',
        b'extern pair abs c-int -- c-int c-int end\nproc main -- do end\n',
        'external procedure may return at most one value',
    ),
    SourceErrorCase(
        'extern_missing_end', b'extern magnitude abs c-int -- c-int\n', 'expected end after external signature'
    ),
    SourceErrorCase('extern_main', b'extern main abs c-int -- c-int end\n', 'main cannot be external'),
    SourceErrorCase(
        'extern_inside_proc',
        b'proc main -- do extern magnitude abs c-int -- c-int end end\n',
        'declarations are only allowed at top level',
    ),
    SourceErrorCase(
        'extern_inside_macro',
        b'macro bad extern magnitude abs c-int -- c-int end end\nproc main -- do end\n',
        'declarations are not allowed in macro bodies',
    ),
    SourceErrorCase(
        'extern_contract_type',
        b'extern magnitude abs c-int -- c-int end\nproc main -- do true magnitude drop end\n',
        'compile-time stack type mismatch',
    ),
    SourceErrorCase(
        'extern_incompatible_contract',
        b'extern as-int abs c-int -- c-int end\nextern as-ptr abs c-int -- c-ptr end\nproc main -- do end\n',
        'incompatible declarations for C symbol',
    ),
    SourceErrorCase(
        'record_wrong_owner',
        b'record Point x int end\nrecord Box x int end\nproc main -- do Box:alloc @Point.x drop end\n',
        'compile-time stack type mismatch',
    ),
    SourceErrorCase(
        'record_wrong_value',
        b'record Point x int end\nproc main -- do Point:alloc let point do true point !Point.x end end\n',
        'compile-time stack type mismatch',
    ),
    SourceErrorCase(
        'record_unknown_field',
        b'record Point x int end\nproc main -- do Point:alloc @Point.y drop end\n',
        'unknown record field',
    ),
    SourceErrorCase(
        'record_legacy_getter',
        b'record Point x int end\nproc main -- do Point:alloc Point.x drop end\n',
        'unknown word',
    ),
    SourceErrorCase(
        'record_legacy_setter',
        b'record Point x int end\nproc main -- do 1 Point:alloc Point.x! end\n',
        'unknown word',
    ),
    SourceErrorCase(
        'record_unknown_type',
        b'record Point x Missing end\nproc main -- do end\n',
        'unknown type in record field',
    ),
    SourceErrorCase(
        'record_duplicate_field',
        b'record Point x int x bool end\nproc main -- do end\n',
        'duplicate record field: x',
    ),
    SourceErrorCase(
        'record_unsupported_cast',
        b'record Point x int end\nproc main -- do 1 Point cast drop end\n',
        'unsupported cast',
    ),
    SourceErrorCase(
        'union_empty', b'union Maybe end\nproc main -- do end\n', 'union must declare at least one variant'
    ),
    SourceErrorCase(
        'union_duplicate_variant',
        b'union Maybe case x case x end\nproc main -- do end\n',
        'duplicate union variant: x',
    ),
    SourceErrorCase(
        'union_unknown_type',
        b'union Maybe case some Missing end\nproc main -- do end\n',
        'unknown type in union variant',
    ),
    SourceErrorCase(
        'union_multiple_payloads',
        b'union Maybe case pair int bool end\nproc main -- do end\n',
        'union variant may carry at most one value',
    ),
    SourceErrorCase(
        'union_unknown_variant',
        b'union Maybe case none end\nproc main -- do Maybe:other drop end\n',
        'unknown union variant',
    ),
    SourceErrorCase(
        'union_wrong_payload_type',
        b'union Maybe case some int end\nproc main -- do true Maybe:some drop end\n',
        'compile-time stack type mismatch',
    ),
    SourceErrorCase(
        'union_unsupported_cast',
        b'union Maybe case none end\nproc main -- do 1 Maybe cast drop end\n',
        'unsupported cast',
    ),
    SourceErrorCase(
        'union_inside_proc',
        b'proc main -- do union Maybe case none end end\n',
        'declarations are only allowed at top level',
    ),
    SourceErrorCase(
        'union_inside_macro',
        b'macro bad union Maybe case none end end\nproc main -- do end\n',
        'declarations are not allowed in macro bodies',
    ),
    SourceErrorCase('union_missing_name', b'union end\nproc main -- do end\n', 'expected union name'),
    SourceErrorCase('union_invalid_name', b'union P.Q case none end\nproc main -- do end\n', 'invalid union name'),
    SourceErrorCase(
        'union_missing_variant_name',
        b'union Maybe case end\nproc main -- do end\n',
        'expected union variant name',
    ),
    SourceErrorCase(
        'union_invalid_variant_name',
        b'union Maybe case P.Q end\nproc main -- do end\n',
        'union variant name must be an identifier',
    ),
    SourceErrorCase(
        'function_contract_mismatch',
        b'fn Mapper int -- int end\nproc nope bool -- int do drop 0 end\nproc main -- do Mapper:ref:nope drop end\n',
        'function reference contract mismatch',
    ),
    SourceErrorCase(
        'function_target_not_found',
        b'fn Mapper -- end\nproc main -- do Mapper:ref:missing drop end\n',
        'function reference target not found',
    ),
    SourceErrorCase(
        'function_unknown_operation',
        b'fn Mapper -- end\nproc main -- do Mapper:nope drop end\n',
        'unknown function operation',
    ),
    SourceErrorCase(
        'function_missing_target',
        b'fn Mapper -- end\nproc main -- do Mapper:ref drop end\n',
        'expected function reference target',
    ),
    SourceErrorCase(
        'function_unknown_signature_type',
        b'fn Mapper Missing -- int end\nproc main -- do end\n',
        'unknown type in function signature',
    ),
    SourceErrorCase('function_missing_name', b'fn -- end\nproc main -- do end\n', 'expected function name'),
    SourceErrorCase('function_invalid_name', b'fn P.Q -- end\nproc main -- do end\n', 'invalid function name'),
    SourceErrorCase(
        'function_missing_separator',
        b'fn Mapper int end\nproc main -- do end\n',
        'expected -- in function signature',
    ),
    SourceErrorCase('function_missing_end', b'fn Mapper -- int', 'expected end after function signature'),
    SourceErrorCase(
        'function_inside_proc', b'proc main -- do fn Mapper -- end end\n', 'declarations are only allowed at top level'
    ),
    SourceErrorCase(
        'function_inside_macro',
        b'macro bad fn Mapper -- end end\nproc main -- do end\n',
        'declarations are not allowed in macro bodies',
    ),
    SourceErrorCase(
        'function_call_type_mismatch',
        b'fn Mapper int -- int end\nproc inc int -- int do 1 + end\nproc main -- do true Mapper:ref:inc Mapper:call drop end\n',
        'compile-time stack type mismatch',
    ),
    SourceErrorCase(
        'function_unsupported_int_cast',
        b'fn Mapper int -- int end\nproc main -- do 1 Mapper cast drop end\n',
        'unsupported cast',
    ),
    SourceErrorCase(
        'function_unsupported_ptr_cast',
        b'fn Mapper int -- int end\nproc main -- do 0 ptr cast Mapper cast drop end\n',
        'unsupported cast',
    ),
    SourceErrorCase(
        'function_unsupported_function_cast',
        b'fn First -- end\nfn Second -- end\nproc target -- do end\nproc main -- do First:ref:target Second cast drop end\n',
        'unsupported cast',
    ),
    SourceErrorCase(
        'function_duplicate_name',
        b'fn Mapper -- end\nfn Mapper -- end\nproc main -- do end\n',
        'duplicate function name: Mapper',
    ),
    SourceErrorCase(
        'function_declaration_collision',
        b'record Mapper value int end\nfn Mapper -- end\nproc main -- do end\n',
        'duplicate declaration name: Mapper',
    ),
    SourceErrorCase(
        'function_output_contract_mismatch',
        b'fn Pair int -- int int end\nproc one int -- int do end\nproc main -- do Pair:ref:one drop end\n',
        'function reference contract mismatch',
    ),
    SourceErrorCase(
        'optimizer_recursive_macro_shadow',
        b'macro + 1 2 + end\nproc main -- do + drop end\n',
        'recursive macro expansion',
    ),
    SourceErrorCase(
        'optimizer_type_error',
        b'proc main -- do 1 true + drop end\n',
        'invalid operand types for pointer/integer arithmetic',
    ),
    SourceErrorCase(
        'reserved-string-record',
        b'record String value int end\nproc main -- do end\n',
        'invalid record name',
        'strings',
    ),
    SourceErrorCase(
        'reserved-string-union', b'union String case value end\nproc main -- do end\n', 'invalid union name', 'strings'
    ),
    SourceErrorCase(
        'reserved-string-function', b'fn String -- end\nproc main -- do end\n', 'invalid function name', 'strings'
    ),
    SourceErrorCase(
        'reserved-string-procedure',
        b'proc String -- do end\n',
        'reserved keyword cannot be a procedure name',
        'strings',
    ),
    SourceErrorCase(
        'reserved-string-macro',
        b'macro String 1 end\nproc main -- do end\n',
        'reserved keyword cannot be a macro name',
        'strings',
    ),
    SourceErrorCase(
        'reserved-string-local',
        b'proc main -- do 1 let String do end end\n',
        'String cannot be a local name',
        'strings',
    ),
    SourceErrorCase(
        'reserved-string-import',
        b'from "lib.frog" import shared as String\nproc main -- do end\n',
        'invalid imported name',
        'strings',
    ),
    SourceErrorCase(
        'reserved-c-string-symbol',
        b'extern bad FrogString -- c-int end\nproc main -- do bad drop end\n',
        'invalid C symbol',
        'strings',
    ),
    SourceErrorCase(
        'string-as-ptr',
        b'proc main -- do "x" @u8 drop end\n',
        'compile-time stack type mismatch',
        'strings',
    ),
    SourceErrorCase(
        'string-as-int',
        b'proc main -- do "x" 1 + drop end\n',
        'invalid operand types for pointer/integer arithmetic',
        'strings',
    ),
]


def run_command(
    arguments: list[str | Path],
    *,
    cwd: Path,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(argument) for argument in arguments],
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


def process_details(result: subprocess.CompletedProcess[bytes]) -> str:
    return f'exit={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}'


def compile_source(
    frogc: Path,
    source: bytes,
    *,
    cwd: Path,
    c_path: Path,
) -> str:
    result = run_command([frogc], cwd=cwd, input_bytes=source)
    assert result.returncode == 0, process_details(result)
    assert result.stderr == b'', process_details(result)
    _ = c_path.write_bytes(result.stdout)
    return result.stdout.decode('utf-8')


def compile_fixture(frogc: Path, cases_root: Path, output_root: Path, case: RunCase) -> tuple[Path, str]:
    source_path = cases_root / case.source
    case_output = output_root / case.group
    case_output.mkdir(parents=True, exist_ok=True)
    c_path = case_output / f'{case.name}.c'
    generated = compile_source(frogc, source_path.read_bytes(), cwd=source_path.parent, c_path=c_path)
    return c_path, generated


def build_executable(root: Path, source_path: Path, fixture: Path, executable: Path) -> None:
    sources = [source_path]
    helper = fixture / 'helper.c'
    if helper.is_file():
        sources.append(helper)

    result = run_command(
        ['gcc', *STRICT_C_FLAGS, *sources, '-o', executable],
        cwd=root,
    )
    assert result.returncode == 0, process_details(result)
    assert result.stdout == b'', process_details(result)
    assert result.stderr == b'', process_details(result)


def run_fixture(
    root: Path,
    frogc: Path,
    cases_root: Path,
    output_root: Path,
    case: RunCase,
) -> str:
    print(f'[REGRESSION:{case.group}] {case.name}')
    c_path, generated = compile_fixture(frogc, cases_root, output_root, case)
    executable = c_path.with_suffix('.exe')
    build_executable(root, c_path, (cases_root / case.source).parent, executable)

    result = run_command([executable, *case.args], cwd=root)
    assert result.returncode == case.expected_status, process_details(result)
    assert result.stderr == b'', process_details(result)
    if case.expected_stdout is not None:
        assert result.stdout == case.expected_stdout, process_details(result)
    return generated


def fixture_error_source(cases_root: Path, case: FixtureErrorCase) -> tuple[Path, Path]:
    fixture = cases_root / case.group / case.name
    if fixture.is_dir():
        return fixture / case.source, fixture

    source = cases_root / case.group / case.source
    return source, source.parent


def assert_compile_error(frogc: Path, source: bytes, *, cwd: Path, expected: str) -> None:
    result = run_command([frogc], cwd=cwd, input_bytes=source)
    assert result.returncode != 0, process_details(result)
    assert result.stderr == f'frogc: {expected}\n'.encode(), process_details(result)


def run_fixture_error(frogc: Path, cases_root: Path, case: FixtureErrorCase) -> None:
    print(f'[REGRESSION:{case.group}-error] {case.name}')
    source, cwd = fixture_error_source(cases_root, case)
    assert_compile_error(frogc, source.read_bytes(), cwd=cwd, expected=case.expected)


def run_source_error(frogc: Path, cases_root: Path, case: SourceErrorCase) -> None:
    print(f'[REGRESSION:{case.cwd}-error] {case.name}')
    assert_compile_error(frogc, case.source, cwd=cases_root / case.cwd, expected=case.expected)


def extract_function(source: str, signature: str) -> str:
    start = source.index(f'{signature} {{\n')
    end = source.index('\n}\n', start) + 3
    return source[start:end]


def assert_semantic_c(generated: dict[str, str]) -> None:
    c_ffi = generated['semantics/c_ffi']
    assert extract_function(
        c_ffi,
        'Cell frog_proc_1_ffi_2Dmix(Cell frog_arg_0, Cell frog_arg_1, Cell frog_arg_2)',
    ) == (
        'Cell frog_proc_1_ffi_2Dmix(Cell frog_arg_0, Cell frog_arg_1, Cell frog_arg_2) {\n'
        '  return (Cell)ffi_test_mix((int)frog_arg_0, (int)(frog_arg_1 != 0), '
        '(void *)(intptr_t)frog_arg_2);\n'
        '}\n'
    )

    functions = generated['semantics/functions_layout']
    for expected in (
        'Cell frog_proc_0_before_2Dinc(void);',
        'void frog_proc_2_unit(void);',
        'Cell frog_proc_3_add(Cell frog_arg_0, Cell frog_arg_1);',
        'frog_results_2 frog_proc_4_duplicate(Cell frog_arg_0);',
        'typedef struct { Cell value_0; Cell value_1; } frog_results_2;',
        '  Cell frog_value_0;',
        '  (void)&frog_value_0;',
        'frog_results_2 frog_call_result = frog_proc_4_duplicate(frog_value_0);',
        'frog_value_1 = frog_call_result.value_1;',
        'frog_value_0 = frog_proc_13_countdown(frog_value_0);',
        'switch (function_id) {',
    ):
        assert expected in functions
    assert 'frog_value_' not in extract_function(functions, 'void frog_proc_2_unit(void)')

    type_stack_growth = generated['semantics/type_stack_growth']
    assert 'Cell frog_value_255;' in type_stack_growth
    assert '(void)&frog_value_255;' in type_stack_growth
    assert 'frog_value_256' not in type_stack_growth

    optimizer = generated['semantics/optimizer_constant_add']
    main_block = extract_function(optimizer, 'void frog_proc_0_main(void)')
    folded_assignments = [
        line
        for line in main_block.splitlines(keepends=True)
        if line in {'  frog_value_0 = 5;\n', '  frog_value_0 = 18;\n', '  frog_value_0 = 9;\n'}
    ]
    assert folded_assignments == [
        '  frog_value_0 = 5;\n',
        '  frog_value_0 = 18;\n',
        '  frog_value_0 = 9;\n',
    ]
    assert (
        '    frog_value_0 = 9223372036854775807;\n'
        '    frog_value_1 = 1;\n'
        '    frog_value_0 = frog_value_0 + frog_value_1;\n'
    ) in main_block

    constants = generated['semantics/constants']
    assert constants.count('= 42;') == 5
    assert 'frog_value_0 = frog_value_0 * frog_value_1;' not in constants

    procedure_symbols = generated['semantics/procedure_symbols']
    for expected in (
        'void frog_proc_0_alpha9(void);',
        'void frog_proc_1_with_5Funder(void);',
        'void frog_proc_2__2B_2B(void);',
        'void frog_proc_3_caf_C3_A9(void);',
        'void frog_proc_4_main(void);',
    ):
        assert expected in procedure_symbols


def string_symbol(source: str, value: str) -> str:
    match = re.search(
        rf'^static uint8_t (frog_string_[0-9_]*)_bytes\[\] = "{re.escape(value)}";$',
        source,
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def assert_string_c(strings: str) -> None:
    lines = strings.splitlines()
    for expected in (
        'typedef struct {',
        '} FrogString;',
        '  uint8_t *bytes;',
        '  Cell len;',
        'static const FrogString frog_string_2771466528 = { frog_string_2771466528_bytes, 8 };',
        'static const FrogString frog_string_2771466528_1 = { frog_string_2771466528_1_bytes, 8 };',
    ):
        assert expected in lines

    for expected in (
        'static uint8_t frog_string_2771466528_bytes[] = ',
        'static uint8_t frog_string_2771466528_1_bytes[] = ',
    ):
        assert any(line.startswith(expected) for line in lines)

    assert len(re.findall(r'^static const FrogString frog_string_', strings, re.MULTILINE)) == 7
    assert len(re.findall(r'^static uint8_t frog_string_', strings, re.MULTILINE)) == 7
    assert strings.count('= "same";') == 1
    same = string_symbol(strings, 'same')
    assert strings.count(f'= (Cell)(intptr_t)&{same};') == 3
    assert sum('frog_alloc(' in line for line in lines) == 1


def assert_stable_string_symbol(frogc: Path, cases_root: Path, output_root: Path) -> None:
    before = compile_source(
        frogc,
        b'proc main -- do "stable" drop end\n',
        cwd=cases_root / 'strings',
        c_path=output_root / 'stable-before.c',
    )
    after = compile_source(
        frogc,
        b'proc main -- do "unrelated" drop "stable" drop end\n',
        cwd=cases_root / 'strings',
        c_path=output_root / 'stable-after.c',
    )
    assert string_symbol(before, 'stable') == string_symbol(after, 'stable')


def assert_large_import_path_error(frogc: Path, cases_root: Path) -> None:
    path = b'a' * 65536
    source = b'from "' + path + b'" import value\nproc main -- do value print end\n'
    assert_compile_error(
        frogc,
        source,
        cwd=cases_root / 'imports',
        expected='import path exceeds max-import-path-bytes',
    )


def run_regressions(root: Path, frogc: Path, cases_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}
    run_cases = [*IMPORT_RUN_CASES, *SEMANTIC_RUN_CASES, *STATUS_CASES, *STRING_RUN_CASES, *STANDALONE_RUN_CASES]
    for run_case in run_cases:
        generated[f'{run_case.group}/{run_case.name}'] = run_fixture(
            root,
            frogc,
            cases_root,
            output_root,
            run_case,
        )

    for fixture_error_case in FIXTURE_ERROR_CASES:
        run_fixture_error(frogc, cases_root, fixture_error_case)
    for source_error_case in SOURCE_ERROR_CASES:
        run_source_error(frogc, cases_root, source_error_case)
    print('[REGRESSION:imports-error] large_path')
    assert_large_import_path_error(frogc, cases_root)

    assert_semantic_c(generated)
    assert_string_c(generated['strings/strings'])
    assert_stable_string_symbol(frogc, cases_root, output_root / 'strings')

    for name, source in generated.items():
        assert not re.search(r'FrogStack|frog_stack|frog_slots|frog_push|frog_pop', source), name
