from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import textwrap
from typing import assert_never


@dataclass(frozen=True)
class SourceSpec:
    body: str = ''
    before_main: str = ''
    after_main: str = ''
    raw_source: str | None = None

    def __post_init__(self) -> None:
        if self.raw_source is not None and (self.body or self.before_main or self.after_main):
            raise ValueError('raw_source cannot be combined with structural source fields')

    def materialize(self) -> str:
        if self.raw_source is not None:
            return self.raw_source

        before_main = textwrap.dedent(self.before_main).strip('\n')
        body = textwrap.dedent(self.body).strip('\n')
        after_main = textwrap.dedent(self.after_main).strip('\n')

        sections: list[str] = []
        if before_main:
            sections.append(before_main)

        main_lines = ['proc main -- do']
        main_lines.extend(f'    {line}' if line else '' for line in body.splitlines())
        main_lines.append('end')
        sections.append('\n'.join(main_lines))

        if after_main:
            sections.append(after_main)
        return '\n\n'.join(sections) + '\n'


type SourceInput = str | SourceSpec


@dataclass(frozen=True)
class CodeExampleGroup:
    name: str
    examples: list[SourceInput]


@dataclass(frozen=True)
class FileCodeExample:
    name: str
    root: SourceSpec
    files: Mapping[str, str]
    main: str = 'main.frog'

    def __post_init__(self) -> None:
        if self.main in self.files:
            raise ValueError('the root source must be provided through root')


@dataclass(frozen=True)
class CliExampleGroup:
    name: str
    examples: list[str]


@dataclass(frozen=True)
class CommandResult:
    command: str
    body: str
    exit_code: int


code_examples: list[SourceInput] = [
    '1 2 + print',
    '1 + print',
    '1 2 +',
    '? 1 ? 2 ? 3 ? 4 ? + ? + ? + ? print ?',
    '+',
    '~',
    '/%',
    #
    '25 7 + print',
    '25 7 - print',
    '25 7 * print',
    '25 7 / print',
    '2 0 / print',
    '25 7 % print',
    '25 7 /% ? print print',
    '25 7 << print',
    '25 7 >> print',
    '25 7 | print',
    '25 7 & print',
    '25 7 ^ print',
    '25 ~ print',
    #
    'true 5 + print',
    'true 5 * print',
    'true 5 % print',
    'true 5 | print',
    'true 5 << print',
    'true 5 && print',
    'true 5 ! print',
    'true 5 swap ~ print',
    'true 5 == print',
    'true 5 < print',
    #
    'true false && print',
    'true false || print',
    'true ! print',
    'false ! print',
    #
    '1 2 == print',
    '1 2 != print',
    '1 2 < print',
    '1 2 > print',
    '1 2 <= print',
    '1 2 >= print',
    #
    '1 2 ? print print',
    '1 dup ? print print',
    '1 2 dup2 ? print print print print',
    '1 2 drop ? print',
    '1 2 swap ? print print',
    '1 2 3 ? print print print',
    '1 2 3 rot ? print print print',
    '1 2 3 4 ? swap2 ? print print print print',
    'print',
    '?',
    'dup',
    'swap',
    'drop',
    'rot',
    #
    '1 2 == if do 5 else 7 end print',
    'if 1 2 == do 5 else 7 end print',
    'if 1 2 != do 5 else 7 end print',
    'if 1 2 != do 5 else 7 8 end print',
    'if 1 2 != do 5 else 7 end',
    'if 1 2 != do 5 else end',
    'if 1 2 != do 5 end',
    'if 1 2 != do 5 print end',
    '5 if 1 2 == do drop 7 end print',
    '5 if 1 2 != do drop 7 end print',
    'if',
    'if 1 2 == do',
    'if 1 2 == do 5 print else',
    'if 1 2 == else 5 end',
    'if do end',
    'if 1 do end',
    'if 1 2 do end',
    'if 1 2 3 do end',
    'if 1 2 3 == do end',
    'else',
    '1 2 if true do swap else end print print',
    'if false do 1 elif true do 2 else 3 end print',
    'if false do 1 elif false do 2 elif true do 3 else 4 end print',
    'if false do 1 print elif true do 2 print end',
    SourceSpec(
        before_main='macro choose if false do 1 elif true do 2 else 3 end end',
        body='choose print',
    ),
    'elif true do 1 end',
    'if false do 1 else 2 elif true do 3 end',
    'if false do 1 elif do 2 end',
    #
    '10 while dup 5 > do 1 - print ? end drop',
    '10 while dup 5 > do 1 - dup print ? end drop',
    '0 while dup 3 < if dup do drop true end do 1 + end print',
    'while else',
    'while end',
    'while',
    'while 1 2 == do',
    'while do end',
    'while 1 do end',
    'while 1 2 do end',
    'while 1 2 3 do end',
    'while 1 2 3 == do end',
    'do',
    'end',
    #
    SourceSpec(before_main='macro dup let x do x x end end', body='1 dup ? print print'),
    SourceSpec(before_main='macro swap let x y do y x end end', body='1 2 swap ? print print'),
    SourceSpec(before_main='macro double dup + end', body='5 double print'),
    SourceSpec(before_main='macro inc 1 + end', body='5 inc print'),
    SourceSpec(body='5 later print', after_main='macro later 1 + end'),
    SourceSpec(before_main='macro choose if 1 2 == do 5 else 7 end end', body='choose print'),
    SourceSpec(before_main='macro one 1 + end\nmacro two one one end', body='5 two print'),
    SourceSpec(before_main='macro loop loop end', body='loop'),
    SourceSpec(before_main='macro a b end\nmacro b a end', body='a'),
    SourceSpec(raw_source='macro outer macro inner 1 end end\n\nproc main -- do\nend\n'),
    SourceSpec(raw_source='macro m else end\n\nproc main -- do\nend\n'),
    SourceSpec(raw_source='macro'),
    SourceSpec(raw_source='macro 123 end\n\nproc main -- do\nend\n'),
    #
    '',
    "'",
    "''",
    "'aa'",
    "'\\n'",
    "'\\t'",
    "'\\''",
    '"',
    '"" ? drop drop',
    '"abc" ? drop drop',
    '"abc\\n" ? drop drop',
    '"abc\'" ? drop drop',
    '"abc\\"" ? drop drop',
    '"A\\n\\x42\\0é" let p n do n print p @u8 print p 1 + @u8 print p 2 + @u8 print p 3 + @u8 print p 4 + @u8 print end',
    '1 // comment\n print',
    #
    SourceSpec(raw_source='proc'),
    'somerandomword',
    #
    SourceSpec(before_main='proc a int -- int do 2 * end', body='5 a print'),
    SourceSpec(before_main='proc a do 2 * end'),
    SourceSpec(before_main='proc a int do 2 * end'),
    SourceSpec(before_main='proc a -- do 2 * end'),
    SourceSpec(before_main='proc a int -- do 2 * end'),
    SourceSpec(before_main='proc a int -- int int do 2 * end'),
    SourceSpec(before_main='proc a int -- int do drop 5 end', body='5 a print'),
    SourceSpec(before_main='proc a bool -- int do drop 5 end', body='5 a print'),
    SourceSpec(before_main='proc a int int -- int do + end', body='5 a print'),
    SourceSpec(before_main='proc a x -- y do + end', body='5 a print'),
    SourceSpec(before_main='proc a int int -- int do + end', body='5 7 a print'),
    SourceSpec(before_main='proc ++ int -- int do 1 + end', body='5 ++ print'),
    #
    SourceSpec(
        before_main='''
        extern magnitude abs c-int -- c-int end
        extern release free c-ptr -- end
        ''',
        body='''
        0 9 - magnitude print
        8 alloc release
        ''',
    ),
    #
    '5 int ? cast ? print',
    '5 bool cast print',
    '0 bool cast print',
    'true int cast print',
    'false int cast print',
    'false bool cast print',
    '1 ptr cast int cast print',
    '9223372036854775807 print',
    '9 alloc let p do 4660 p 1 + !u16 p 1 + @u16 print end',
    SourceSpec(
        before_main='proc forward -- int do later end\nproc later -- int do 42 end',
        body='forward print',
    ),
    SourceSpec(before_main='proc sink int -- do drop end', body='1 sink'),
    SourceSpec(
        before_main='''
        proc countdown int -- int do
            if dup 0 == do drop 0 else 1 - countdown end
        end
        ''',
        body='3 countdown print',
    ),
    '4 alloc let p do 42 p !i8 p @i8 print end',
    '4 alloc let p do 255 p !u8 p @u8 print p @i8 print end',
    '4 alloc let p do 4660 p !u16 p @u16 print p 1 + @u8 print end',
    '4 alloc let p do 127 p !i8 p @i8 print 128 p !u8 p @u8 print end',
    SourceSpec(
        before_main='proc cell ptr int -- int do + @u8 end',
        body='4 alloc let p do 42 p !u8 p 0 cell print end',
    ),
    '"README.md" read-file let data length success do success print length 0 > print data @u8 putc 10 putc end',
    '"frog-read-file-missing" read-file let data length success do data drop length print success print end',
    'args let argv argc do argc print argv @ptr @u8 putc 10 putc end',
    SourceSpec(raw_source=''),
    SourceSpec(raw_source='1 print\n'),
]

code_example_groups = [
    CodeExampleGroup('basics', code_examples[0:7]),
    CodeExampleGroup('arithmetic', code_examples[7:20]),
    CodeExampleGroup('int_bool_type_errors', code_examples[20:30]),
    CodeExampleGroup('booleans', code_examples[30:34]),
    CodeExampleGroup('comparisons', code_examples[34:40]),
    CodeExampleGroup('stack_intrinsics', code_examples[40:54]),
    CodeExampleGroup('if_blocks', code_examples[54:82]),
    CodeExampleGroup('while_blocks', code_examples[82:96]),
    CodeExampleGroup('macros', code_examples[96:109]),
    CodeExampleGroup('literals_and_comments', code_examples[109:124]),
    CodeExampleGroup('words', code_examples[124:126]),
    CodeExampleGroup('procedures', code_examples[126:138]),
    CodeExampleGroup('c_ffi', code_examples[138:139]),
    CodeExampleGroup('casts_and_memory', code_examples[139:158]),
    CodeExampleGroup('process_arguments', code_examples[158:159]),
    CodeExampleGroup('program_structure', code_examples[159:161]),
]

assert sum(len(group.examples) for group in code_example_groups) == len(code_examples)
assert len(code_examples) == 161

file_code_examples = [
    FileCodeExample(
        name='import_proc',
        root=SourceSpec(before_main='from "math.frog" import inc', body='41 inc print'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='import_group',
        root=SourceSpec(
            before_main='from "math.frog" import ( inc dec add2 )',
            body='''
            5 inc print
            5 dec print
            5 add2 print
            ''',
        ),
        files={
            'math.frog': '''
            proc inc int -- int do 1 + end
            proc dec int -- int do 1 - end
            proc add2 int -- int do 2 + end
            ''',
        },
    ),
    FileCodeExample(
        name='import_alias',
        root=SourceSpec(before_main='from "math.frog" import inc as bump', body='1 bump print'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='same_import_twice_is_ok',
        root=SourceSpec(
            before_main='''
            from "math.frog" import inc
            from "math.frog" import inc
            ''',
            body='1 inc print',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='same_import_two_aliases',
        root=SourceSpec(
            before_main='''
            from "math.frog" import inc
            from "math.frog" import inc as bump
            ''',
            body='''
            1 inc print
            1 bump print
            ''',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='use_before_import_declaration_should_work',
        root=SourceSpec(
            body='''
            // imports are collected before bodies are compiled
            10 inc print
            ''',
            after_main='from "math.frog" import inc',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='import_paths_are_root_relative',
        root=SourceSpec(before_main='from "pkg/use.frog" import value', body='value print'),
        files={
            'math.frog': 'proc value -- int do 999 end\n',
            'pkg/math.frog': 'proc value -- int do 7 end\n',
            'pkg/use.frog': '''
            // "math.frog" resolves from the root file directory, not from pkg/
            from "math.frog" import value as root_value

            proc value -- int do root_value end
            ''',
        },
    ),
    FileCodeExample(
        name='explicit_subdir_import_path',
        root=SourceSpec(before_main='from "pkg/use.frog" import value', body='value print'),
        files={
            'math.frog': 'proc value -- int do 999 end\n',
            'pkg/math.frog': 'proc value -- int do 7 end\n',
            'pkg/use.frog': '''
            from "pkg/math.frog" import value as pkg_value

            proc value -- int do pkg_value end
            ''',
        },
    ),
    FileCodeExample(
        name='canonical_import_paths_share_module',
        root=SourceSpec(
            before_main='''
            from "lib/math.frog" import value
            from "lib/../lib/math.frog" import value
            ''',
            body='value print',
        ),
        files={
            'lib/math.frog': 'proc value -- int do 42 end\n',
        },
    ),
    FileCodeExample(
        name='imported_top_level_code_does_not_run',
        root=SourceSpec(before_main='from "lib.frog" import value', body='value print'),
        files={
            'lib.frog': '''
            99 print

            proc value -- int do 7 end
            ''',
        },
    ),
    FileCodeExample(
        name='imported_proc_uses_own_module_scope',
        root=SourceSpec(
            before_main='''
            from "lib.frog" import value

            proc helper -- int do 99 end
            ''',
            body='''
            value print
            helper print
            ''',
        ),
        files={
            'lib.frog': '''
            proc helper -- int do 7 end
            proc value -- int do helper end
            ''',
        },
    ),
    FileCodeExample(
        name='same_private_name_in_two_imported_modules',
        root=SourceSpec(
            before_main='''
            from "left.frog" import value as left
            from "right.frog" import value as right
            ''',
            body='''
            left print
            right print
            ''',
        ),
        files={
            'left.frog': '''
            proc helper -- int do 10 end
            proc value -- int do helper end
            ''',
            'right.frog': '''
            proc helper -- int do 20 end
            proc value -- int do helper end
            ''',
        },
    ),
    FileCodeExample(
        name='local_proc_uses_imported_proc',
        root=SourceSpec(
            before_main='''
            from "math.frog" import inc

            proc add_two int -- int do inc inc end
            ''',
            body='3 add_two print',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='local_macro_uses_imported_proc',
        root=SourceSpec(
            before_main='''
            from "math.frog" import inc

            macro add_two inc inc end
            ''',
            body='3 add_two print',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reexport_proc',
        root=SourceSpec(before_main='from "facade.frog" import inc', body='4 inc print'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': 'from "math.frog" import inc\n',
        },
    ),
    FileCodeExample(
        name='reexport_alias',
        root=SourceSpec(before_main='from "facade.frog" import bump', body='4 bump print'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': 'from "math.frog" import inc as bump\n',
        },
    ),
    FileCodeExample(
        name='module_uses_imported_proc',
        root=SourceSpec(before_main='from "facade.frog" import add_two', body='4 add_two print'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': '''
            from "math.frog" import inc

            proc add_two int -- int do inc inc end
            ''',
        },
    ),
    FileCodeExample(
        name='diamond_reexports',
        root=SourceSpec(
            before_main='''
            from "left.frog" import value as left
            from "right.frog" import value as right
            ''',
            body='''
            left print
            right print
            ''',
        ),
        files={
            'base.frog': 'proc inc int -- int do 1 + end\n',
            'left.frog': '''
            from "base.frog" import inc

            proc value -- int do 10 inc end
            ''',
            'right.frog': '''
            from "base.frog" import inc

            proc value -- int do 20 inc end
            ''',
        },
    ),
    FileCodeExample(
        name='import_macro',
        root=SourceSpec(before_main='from "macros.frog" import twice', body='21 twice print'),
        files={
            'macros.frog': 'macro twice dup + end\n',
        },
    ),
    FileCodeExample(
        name='imported_macro_uses_defining_module_proc',
        root=SourceSpec(
            before_main='''
            from "macros.frog" import use_secret

            proc secret int -- int do 1 + end
            ''',
            body='''
            5 use_secret print
            5 secret print
            ''',
        ),
        files={
            'macros.frog': '''
            proc secret int -- int do 100 + end
            macro use_secret secret end
            ''',
        },
    ),
    FileCodeExample(
        name='imported_macro_uses_defining_module_macro',
        root=SourceSpec(before_main='from "macros.frog" import add_two', body='5 add_two print'),
        files={
            'macros.frog': '''
            macro inc 1 + end
            macro add_two inc inc end
            ''',
        },
    ),
    FileCodeExample(
        name='imported_macro_uses_defining_module_import',
        root=SourceSpec(before_main='from "facade.frog" import bump', body='5 bump print'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': '''
            from "math.frog" import inc

            macro bump inc end
            ''',
        },
    ),
    FileCodeExample(
        name='reexported_macro_keeps_original_scope',
        root=SourceSpec(
            before_main='''
            from "facade.frog" import bump

            proc helper int -- int do 1 + end
            ''',
            body='''
            5 bump print
            5 helper print
            ''',
        ),
        files={
            'macros.frog': '''
            proc helper int -- int do 10 + end
            macro bump helper end
            ''',
            'facade.frog': 'from "macros.frog" import bump\n',
        },
    ),
    FileCodeExample(
        name='imported_macro_with_blocks',
        root=SourceSpec(
            before_main='from "macros.frog" import move_away_from_zero',
            body='''
            5 move_away_from_zero print
            0 move_away_from_zero print
            ''',
        ),
        files={
            'macros.frog': 'macro move_away_from_zero if dup 0 > do 1 + else 1 - end end\n',
        },
    ),
    FileCodeExample(
        name='imported_macro_with_let',
        root=SourceSpec(
            before_main='from "macros.frog" import over',
            body='1 2 over print print print',
        ),
        files={
            'macros.frog': 'macro over let a b do a b a end end\n',
        },
    ),
    FileCodeExample(
        name='missing_imported_file',
        root=SourceSpec(before_main='from "missing.frog" import inc'),
        files={},
    ),
    FileCodeExample(
        name='missing_imported_name',
        root=SourceSpec(before_main='from "math.frog" import inc'),
        files={
            'math.frog': 'proc dec int -- int do 1 - end\n',
        },
    ),
    FileCodeExample(
        name='alias_does_not_bind_original_name',
        root=SourceSpec(before_main='from "math.frog" import inc as bump', body='1 inc print'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_import_then_local_proc',
        root=SourceSpec(
            before_main='''
            from "math.frog" import inc

            proc inc int -- int do 2 + end
            ''',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_local_proc_then_import',
        root=SourceSpec(
            before_main='proc inc int -- int do 2 + end',
            after_main='from "math.frog" import inc',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_import_then_local_macro',
        root=SourceSpec(
            before_main='''
            from "math.frog" import inc

            macro inc 2 + end
            ''',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_two_imports_same_name',
        root=SourceSpec(
            before_main='''
            from "left.frog" import value
            from "right.frog" import value
            ''',
        ),
        files={
            'left.frog': 'proc value -- int do 1 end\n',
            'right.frog': 'proc value -- int do 2 end\n',
        },
    ),
    FileCodeExample(
        name='conflict_two_imports_same_alias',
        root=SourceSpec(
            before_main='''
            from "left.frog" import value as shared
            from "right.frog" import other as shared
            ''',
        ),
        files={
            'left.frog': 'proc value -- int do 1 end\n',
            'right.frog': 'proc other -- int do 2 end\n',
        },
    ),
    FileCodeExample(
        name='direct_import_cycle',
        root=SourceSpec(before_main='from "a.frog" import value'),
        files={
            'a.frog': 'from "b.frog" import value\n',
            'b.frog': 'from "a.frog" import value\n',
        },
    ),
    FileCodeExample(
        name='self_import_cycle',
        root=SourceSpec(before_main='from "a.frog" import value'),
        files={
            'a.frog': 'from "a.frog" import value\n',
        },
    ),
    FileCodeExample(
        name='reject_wildcard_import',
        root=SourceSpec(before_main='from "math.frog" import *'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reject_comma_in_group_import',
        root=SourceSpec(before_main='from "math.frog" import ( inc , dec )'),
        files={
            'math.frog': '''
            proc inc int -- int do 1 + end
            proc dec int -- int do 1 - end
            ''',
        },
    ),
    FileCodeExample(
        name='reject_module_alias_form_for_now',
        root=SourceSpec(before_main='import "math.frog" as math'),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reject_import_inside_proc',
        root=SourceSpec(
            raw_source='proc main -- do\n    from "math.frog" import inc\nend\n',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reject_import_inside_macro',
        root=SourceSpec(
            raw_source='''macro bad from "math.frog" import inc end

proc main -- do
    bad
end
''',
        ),
        files={
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
]
cli_example_groups = [
    CliExampleGroup(
        'usage',
        [
            '-h',
            '--help',
            'run test/tmp_fs/missing.frog',
            '--unknown',
            'unknown',
            'build -o build/frogc examples/01_simple.frog',
        ],
    ),
    CliExampleGroup(
        'run_code',
        ['run -c "proc main -- do 42 print end"'],
    ),
]

assert len(file_code_examples) == 40

ROOT = Path(__file__).parent.parent
FROGC = ROOT / 'build' / 'frogc'

dir_examples = ROOT / 'examples'
dir_tests = ROOT / 'test'
dir_snapshots = dir_tests / 'snapshots'
tmp_fs = dir_tests / 'tmp_fs'
COMMAND_TIMEOUT_SECONDS = 30
COMMAND_TERMINATION_GRACE_SECONDS = 2


def as_source_spec(source: SourceInput) -> SourceSpec:
    match source:
        case str():
            return SourceSpec(body=source)
        case SourceSpec():
            return source
        case _:
            assert_never(source)


def materialize_source(source: SourceInput) -> str:
    return as_source_spec(source).materialize()


def assert_structural_main(source: SourceInput) -> None:
    spec = as_source_spec(source)
    if spec.raw_source is not None:
        return
    assert '\nproc main -- do\n' in f'\n{spec.materialize()}'


def ensure_trailing_newline(text: str) -> str:
    if text.endswith('\n'):
        return text
    return f'{text}\n'


def source_fence(text: str) -> str:
    return f'```frog\n{ensure_trailing_newline(text)}```\n'


def output_fence(text: str) -> str:
    if text == '':
        text = '(no output)\n'
    return f'```text\n{ensure_trailing_newline(text)}```\n'


def render_source(label: str, text: str) -> str:
    return f'### Source: `{label}`\n\n{source_fence(text)}\n'


def capture_frog(*args: str | Path, env: Mapping[str, str] | None = None) -> CommandResult:
    command_args = [str(arg) for arg in args]
    command = shlex.join(['build/frogc', *command_args])
    process = subprocess.Popen(
        [FROGC, *command_args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        env=env,
        start_new_session=True,
    )
    try:
        body, _ = process.communicate(timeout=COMMAND_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            body, _ = process.communicate(timeout=COMMAND_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            body, _ = process.communicate()
        details = f'\n{body}' if body else ''
        raise TimeoutError(f'{command} exceeded {COMMAND_TIMEOUT_SECONDS} seconds{details}') from None
    assert process.returncode is not None
    return CommandResult(command, body, process.returncode)


def render_result(title: str, result: CommandResult) -> str:
    body = result.body
    if result.exit_code != 0:
        body = f'{ensure_trailing_newline(body)}[EXIT CODE] {result.exit_code}\n'
    return f'### {title}\n\nCommand:\n\n{output_fence(result.command)}\nOutput:\n\n{output_fence(body)}\n'


def render_cli_sources(args: list[str]) -> str:
    rendered: list[str] = []
    for arg in args:
        file = ROOT / arg
        if file.is_file() and file.suffix == '.frog':
            rendered.append(render_source(file.relative_to(ROOT).as_posix(), file.read_text()))

    return ''.join(rendered)


def write_snapshot(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text)


def snapshot_header(name: str) -> str:
    return f'# Snapshot: {name}\n\n'


for code_example in code_examples:
    assert_structural_main(code_example)
for file_code_example in file_code_examples:
    assert_structural_main(file_code_example.root)


shutil.rmtree(dir_snapshots, ignore_errors=True)
dir_snapshots.mkdir(parents=True)

try:
    shutil.rmtree(tmp_fs, ignore_errors=True)
    tmp_fs.mkdir(parents=True)

    for file_example in sorted(dir_examples.iterdir()):
        if not file_example.is_file():
            continue
        if file_example.suffix != '.frog':
            continue

        relative_file = file_example.relative_to(ROOT)
        print(f'[FILE] {relative_file}')
        run_result = capture_frog('run', relative_file)

        snapshot_name = relative_file.with_suffix('').as_posix()
        write_snapshot(
            dir_snapshots / 'examples' / relative_file.with_suffix('.out').name,
            ''.join(
                [
                    snapshot_header(snapshot_name),
                    render_source(relative_file.as_posix(), file_example.read_text()),
                    render_result('Run', run_result),
                ]
            ),
        )

    for cli_group in cli_example_groups:
        parts = [snapshot_header(f'cli/{cli_group.name}')]
        for cli_example in cli_group.examples:
            print(f'[CLI:{cli_group.name}] {cli_example}')
            args = shlex.split(cli_example)
            result = capture_frog(*args)
            parts.append(f'## Case: `{cli_example}`\n\n')
            parts.append(render_cli_sources(args))
            parts.append(render_result('Result', result))

        write_snapshot(dir_snapshots / 'cli' / f'{cli_group.name}.out', ''.join(parts))

    build_case = tmp_fs / 'build_run'
    build_case.mkdir()
    build_source = SourceSpec(body='40 2 + print').materialize()
    build_main = build_case / 'main.frog'
    _ = build_main.write_text(build_source)
    build_main_relative = build_main.relative_to(ROOT)
    print(f'[CLI:build_run] {build_main_relative}')
    build_result = capture_frog('build', '-r', build_main_relative)
    write_snapshot(
        dir_snapshots / 'cli' / 'build_run.out',
        ''.join(
            [
                snapshot_header('cli/build_run'),
                render_source(build_main_relative.as_posix(), build_source),
                render_result('Build and Run', build_result),
            ]
        ),
    )

    assert build_result.exit_code == 0
    built_c = build_main.with_suffix('.c')
    built_executable = build_main.with_suffix('.exe')
    original_c = built_c.read_bytes()
    original_executable = built_executable.read_bytes()

    _ = build_main.write_text(SourceSpec(body='43 print').materialize())
    fake_bin = build_case / 'fake-bin'
    fake_bin.mkdir()
    false_executable = shutil.which('false')
    assert false_executable is not None
    (fake_bin / 'gcc').symlink_to(false_executable)
    failing_environment = os.environ.copy()
    assert 'PATH' in failing_environment
    failing_environment['PATH'] = f'{fake_bin}{os.pathsep}{failing_environment["PATH"]}'
    failed_build = capture_frog('build', build_main_relative, env=failing_environment)
    assert failed_build.exit_code != 0
    failed_c = built_c.read_bytes()
    assert failed_c != original_c
    assert built_executable.read_bytes() == original_executable

    updated_build = capture_frog('build', build_main_relative)
    assert updated_build.exit_code == 0
    assert built_c.read_bytes() == failed_c
    assert built_executable.read_bytes() != original_executable

    symlink_case = tmp_fs / 'symlink_root'
    real_directory = symlink_case / 'real'
    lexical_directory = symlink_case / 'lexical'
    real_directory.mkdir(parents=True)
    lexical_directory.mkdir()
    real_main = real_directory / 'main.frog'
    _ = real_main.write_text(SourceSpec(before_main='from "value.frog" import value', body='value print').materialize())
    _ = (real_directory / 'value.frog').write_text('proc value -- int do 99 end\n')
    _ = (lexical_directory / 'value.frog').write_text('proc value -- int do 42 end\n')
    lexical_main = lexical_directory / 'main.frog'
    lexical_main.symlink_to(real_main)
    symlink_result = capture_frog('run', lexical_main.relative_to(ROOT))
    assert symlink_result.exit_code == 0
    assert symlink_result.body == '42\n'

    for code_group in code_example_groups:
        parts = [snapshot_header(f'code/{code_group.name}')]
        for index, code_example in enumerate(code_group.examples, start=1):
            print(f'[CODE:{code_group.name}] {index}: {code_example!r}')
            materialized = materialize_source(code_example)
            tmp = tmp_fs / 'code.frog'
            _ = tmp.write_text(materialized)
            relative_tmp = tmp.relative_to(ROOT)

            run_result = capture_frog('run', relative_tmp)

            parts.append(f'## Case {index:02d}\n\n')
            parts.append(render_source(relative_tmp.as_posix(), materialized))
            parts.append(render_result('Run', run_result))

        write_snapshot(dir_snapshots / 'code' / f'{code_group.name}.out', ''.join(parts))

    for file_code_example in file_code_examples:
        print(f'[FILES] {file_code_example.name}')
        tmp_fs_case = tmp_fs / file_code_example.name
        tmp_fs_case.mkdir(parents=True)
        main_file = tmp_fs_case / file_code_example.main
        main_source = file_code_example.root.materialize()
        main_file.parent.mkdir(parents=True, exist_ok=True)
        _ = main_file.write_text(main_source)
        for file_name, text in file_code_example.files.items():
            file = tmp_fs_case / file_name
            file.parent.mkdir(parents=True, exist_ok=True)
            _ = file.write_text(text)

        main = main_file.relative_to(ROOT)
        run_result = capture_frog('run', main)

        parts = [snapshot_header(f'imports/{file_code_example.name}')]
        parts.append(render_source(main.as_posix(), main_source))
        for file_name, text in file_code_example.files.items():
            parts.append(render_source((tmp_fs_case / file_name).relative_to(ROOT).as_posix(), text))

        parts.append(f'### Main: `{main.as_posix()}`\n\n')
        parts.append(render_result('Run', run_result))
        write_snapshot(dir_snapshots / 'imports' / f'{file_code_example.name}.out', ''.join(parts))
finally:
    shutil.rmtree(tmp_fs, ignore_errors=True)
