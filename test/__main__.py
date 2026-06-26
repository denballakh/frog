import contextlib
from dataclasses import dataclass
from pathlib import Path
import io
import shutil
import shlex

from frog.__main__ import run_frog


@dataclass(frozen=True)
class CodeExampleGroup:
    name: str
    examples: list[str]


@dataclass(frozen=True)
class FileCodeExample:
    name: str
    files: dict[str, str]
    main: str = 'main.frog'


@dataclass(frozen=True)
class CliExampleGroup:
    name: str
    examples: list[str]


@dataclass(frozen=True)
class CommandResult:
    command: str
    body: str
    exit_code: int


code_examples = [
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
    #
    '10 while dup 5 > do 1 - print ? end drop',
    '10 while dup 5 > do 1 - dup print ? end drop',
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
    'macro dup let x do x x end end 1 dup ? print print',
    'macro swap let x y do y x end end 1 2 swap ? print print',
    'macro double dup + end 5 double print',
    'macro inc 1 + end 5 inc print',
    '5 later print macro later 1 + end',
    'macro choose if 1 2 == do 5 else 7 end end choose print',
    'macro one 1 + end macro two one one end 5 two print',
    'macro loop loop end loop',
    'macro a b end macro b a end a',
    'macro outer macro inner 1 end end',
    'macro m else end',
    'macro',
    'macro 123 end',
    #
    '',
    "'",
    "''",
    "'aa'",
    "'\\n'",
    "'\\t'",
    "'\\''",
    '"',
    '"" ?',
    '"abc" ?',
    '"abc\\n" ?',
    '"abc\'" ?',
    '"abc\\"" ?',
    '1 // comment \n print',
    #
    'proc',
    'somerandomword',
    #
    '''
    proc a int -- int do 2 * end
    5 a print
    ''',
    '''
    proc a do 2 * end
    ''',
    '''
    proc a int do 2 * end
    ''',
    '''
    proc a -- do 2 * end
    ''',
    '''
    proc a int -- do 2 * end
    ''',
    '''
    proc a int -- int int do 2 * end
    ''',
    '''
    proc a int -- int do drop 5 end
    5 a print
    ''',
    '''
    proc a bool -- int do drop 5 end
    5 a print
    ''',
    '''
    proc a int int -- int do + end
    5 a print
    ''',
    '''
    proc a x -- y do + end
    5 a print
    ''',
    '''
    proc a int int -- int do + end
    5 7 a print
    ''',
    '''
    proc ++ int -- int do 1 + end
    5 ++ print
    ''',
    #
    '5 int ? cast ? print',
    '5 bool cast print',
    '0 bool cast print',
    'true int cast print',
    'false int cast print',
    'false bool cast print',
    '1 ptr cast int cast print',
    '4 alloc let p do 42 p !i8 p @i8 print end',
    '4 alloc let p do 255 p !u8 p @u8 print p @i8 print end',
    '4 alloc let p do 4660 p !u16 p @u16 print p 1 + @u8 print end',
    '4 alloc let p do 127 p !i8 p @i8 print 128 p !u8 p @u8 print end',
    '''
    proc cell ptr int -- int do + @u8 end
    4 alloc let p do 42 p !u8 p 0 cell print end
    ''',
]

code_example_groups = [
    CodeExampleGroup('basics', code_examples[0:7]),
    CodeExampleGroup('arithmetic', code_examples[7:20]),
    CodeExampleGroup('int_bool_type_errors', code_examples[20:30]),
    CodeExampleGroup('booleans', code_examples[30:34]),
    CodeExampleGroup('comparisons', code_examples[34:40]),
    CodeExampleGroup('stack_intrinsics', code_examples[40:54]),
    CodeExampleGroup('if_blocks', code_examples[54:74]),
    CodeExampleGroup('while_blocks', code_examples[74:87]),
    CodeExampleGroup('macros', code_examples[87:100]),
    CodeExampleGroup('literals_and_comments', code_examples[100:114]),
    CodeExampleGroup('words', code_examples[114:116]),
    CodeExampleGroup('procedures', code_examples[116:128]),
    CodeExampleGroup('casts_and_memory', code_examples[128:140]),
]

assert sum(len(group.examples) for group in code_example_groups) == len(code_examples)

file_code_examples = [
    FileCodeExample(
        name='import_proc',
        files={
            'main.frog': '''
            from "math.frog" import inc

            41 inc print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='import_group',
        files={
            'main.frog': '''
            from "math.frog" import ( inc dec add2 )

            5 inc print
            5 dec print
            5 add2 print
            ''',
            'math.frog': '''
            proc inc int -- int do 1 + end
            proc dec int -- int do 1 - end
            proc add2 int -- int do 2 + end
            ''',
        },
    ),
    FileCodeExample(
        name='import_alias',
        files={
            'main.frog': '''
            from "math.frog" import inc as bump

            1 bump print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='same_import_twice_is_ok',
        files={
            'main.frog': '''
            from "math.frog" import inc
            from "math.frog" import inc

            1 inc print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='same_import_two_aliases',
        files={
            'main.frog': '''
            from "math.frog" import inc
            from "math.frog" import inc as bump

            1 inc print
            1 bump print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='use_before_import_declaration_should_work',
        files={
            'main.frog': '''
            // imports are collected before bodies are compiled
            10 inc print

            from "math.frog" import inc
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='import_paths_are_root_relative',
        files={
            'main.frog': '''
            from "pkg/use.frog" import value

            value print
            ''',
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
        files={
            'main.frog': '''
            from "pkg/use.frog" import value

            value print
            ''',
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
        files={
            'main.frog': '''
            from "lib/math.frog" import value
            from "lib/../lib/math.frog" import value

            value print
            ''',
            'lib/math.frog': 'proc value -- int do 42 end\n',
        },
    ),
    FileCodeExample(
        name='imported_top_level_code_does_not_run',
        files={
            'main.frog': '''
            from "lib.frog" import value

            value print
            ''',
            'lib.frog': '''
            99 print

            proc value -- int do 7 end
            ''',
        },
    ),
    FileCodeExample(
        name='imported_proc_uses_own_module_scope',
        files={
            'main.frog': '''
            from "lib.frog" import value

            proc helper -- int do 99 end

            value print
            helper print
            ''',
            'lib.frog': '''
            proc helper -- int do 7 end
            proc value -- int do helper end
            ''',
        },
    ),
    FileCodeExample(
        name='same_private_name_in_two_imported_modules',
        files={
            'main.frog': '''
            from "left.frog" import value as left
            from "right.frog" import value as right

            left print
            right print
            ''',
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
        files={
            'main.frog': '''
            from "math.frog" import inc

            proc add_two int -- int do inc inc end

            3 add_two print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='local_macro_uses_imported_proc',
        files={
            'main.frog': '''
            from "math.frog" import inc

            macro add_two inc inc end

            3 add_two print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reexport_proc',
        files={
            'main.frog': '''
            from "facade.frog" import inc

            4 inc print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': 'from "math.frog" import inc\n',
        },
    ),
    FileCodeExample(
        name='reexport_alias',
        files={
            'main.frog': '''
            from "facade.frog" import bump

            4 bump print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': 'from "math.frog" import inc as bump\n',
        },
    ),
    FileCodeExample(
        name='module_uses_imported_proc',
        files={
            'main.frog': '''
            from "facade.frog" import add_two

            4 add_two print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': '''
            from "math.frog" import inc

            proc add_two int -- int do inc inc end
            ''',
        },
    ),
    FileCodeExample(
        name='diamond_reexports',
        files={
            'main.frog': '''
            from "left.frog" import value as left
            from "right.frog" import value as right

            left print
            right print
            ''',
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
        files={
            'main.frog': '''
            from "macros.frog" import twice

            21 twice print
            ''',
            'macros.frog': 'macro twice dup + end\n',
        },
    ),
    FileCodeExample(
        name='imported_macro_uses_defining_module_proc',
        files={
            'main.frog': '''
            from "macros.frog" import use_secret

            proc secret int -- int do 1 + end

            5 use_secret print
            5 secret print
            ''',
            'macros.frog': '''
            proc secret int -- int do 100 + end
            macro use_secret secret end
            ''',
        },
    ),
    FileCodeExample(
        name='imported_macro_uses_defining_module_macro',
        files={
            'main.frog': '''
            from "macros.frog" import add_two

            5 add_two print
            ''',
            'macros.frog': '''
            macro inc 1 + end
            macro add_two inc inc end
            ''',
        },
    ),
    FileCodeExample(
        name='imported_macro_uses_defining_module_import',
        files={
            'main.frog': '''
            from "facade.frog" import bump

            5 bump print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
            'facade.frog': '''
            from "math.frog" import inc

            macro bump inc end
            ''',
        },
    ),
    FileCodeExample(
        name='reexported_macro_keeps_original_scope',
        files={
            'main.frog': '''
            from "facade.frog" import bump

            proc helper int -- int do 1 + end

            5 bump print
            5 helper print
            ''',
            'macros.frog': '''
            proc helper int -- int do 10 + end
            macro bump helper end
            ''',
            'facade.frog': 'from "macros.frog" import bump\n',
        },
    ),
    FileCodeExample(
        name='imported_macro_with_blocks',
        files={
            'main.frog': '''
            from "macros.frog" import move_away_from_zero

            5 move_away_from_zero print
            0 move_away_from_zero print
            ''',
            'macros.frog': 'macro move_away_from_zero if dup 0 > do 1 + else 1 - end end\n',
        },
    ),
    FileCodeExample(
        name='imported_macro_with_let',
        files={
            'main.frog': '''
            from "macros.frog" import over

            1 2 over print print print
            ''',
            'macros.frog': 'macro over let a b do a b a end end\n',
        },
    ),
    FileCodeExample(
        name='missing_imported_file',
        files={
            'main.frog': 'from "missing.frog" import inc\n',
        },
    ),
    FileCodeExample(
        name='missing_imported_name',
        files={
            'main.frog': 'from "math.frog" import inc\n',
            'math.frog': 'proc dec int -- int do 1 - end\n',
        },
    ),
    FileCodeExample(
        name='alias_does_not_bind_original_name',
        files={
            'main.frog': '''
            from "math.frog" import inc as bump

            1 inc print
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_import_then_local_proc',
        files={
            'main.frog': '''
            from "math.frog" import inc

            proc inc int -- int do 2 + end
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_local_proc_then_import',
        files={
            'main.frog': '''
            proc inc int -- int do 2 + end

            from "math.frog" import inc
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_import_then_local_macro',
        files={
            'main.frog': '''
            from "math.frog" import inc

            macro inc 2 + end
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='conflict_two_imports_same_name',
        files={
            'main.frog': '''
            from "left.frog" import value
            from "right.frog" import value
            ''',
            'left.frog': 'proc value -- int do 1 end\n',
            'right.frog': 'proc value -- int do 2 end\n',
        },
    ),
    FileCodeExample(
        name='conflict_two_imports_same_alias',
        files={
            'main.frog': '''
            from "left.frog" import value as shared
            from "right.frog" import other as shared
            ''',
            'left.frog': 'proc value -- int do 1 end\n',
            'right.frog': 'proc other -- int do 2 end\n',
        },
    ),
    FileCodeExample(
        name='direct_import_cycle',
        files={
            'main.frog': 'from "a.frog" import value\n',
            'a.frog': 'from "b.frog" import value\n',
            'b.frog': 'from "a.frog" import value\n',
        },
    ),
    FileCodeExample(
        name='self_import_cycle',
        files={
            'main.frog': 'from "a.frog" import value\n',
            'a.frog': 'from "a.frog" import value\n',
        },
    ),
    FileCodeExample(
        name='reject_wildcard_import',
        files={
            'main.frog': 'from "math.frog" import *\n',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reject_comma_in_group_import',
        files={
            'main.frog': 'from "math.frog" import ( inc , dec )\n',
            'math.frog': '''
            proc inc int -- int do 1 + end
            proc dec int -- int do 1 - end
            ''',
        },
    ),
    FileCodeExample(
        name='reject_module_alias_form_for_now',
        files={
            'main.frog': 'import "math.frog" as math\n',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reject_import_inside_proc',
        files={
            'main.frog': '''
            proc main -- do
                from "math.frog" import inc
            end
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
    FileCodeExample(
        name='reject_import_inside_macro',
        files={
            'main.frog': '''
            macro bad from "math.frog" import inc end
            bad
            ''',
            'math.frog': 'proc inc int -- int do 1 + end\n',
        },
    ),
]
cli_example_groups = [
    CliExampleGroup(
        'usage_errors',
        [
            '',
            '-h',
            '--help',
            'run',
            'run xxx',
            '-l',
            '-l TRACE',
            '-l TRACE run',
        ],
    ),
    CliExampleGroup(
        'log_levels',
        [
            '-l TRACE run examples/01_simple.frog',
            '-l LOL run examples/01_simple.frog',
            '-l WARN run examples/01_simple.frog',
            '-l INFO run examples/01_simple.frog',
            '-l ERROR run examples/01_simple.frog',
        ],
    ),
    CliExampleGroup(
        'trace_examples',
        [
            '-l TRACE run examples/02_while.frog',
            '-l TRACE build -r examples/02_while.frog',
        ],
    ),
]

ROOT = Path(__file__).parent.parent

dir_examples = ROOT / 'examples'
dir_tests = ROOT / 'test'
dir_snapshots = dir_tests / 'snapshots'
tmp_fs = dir_tests / 'tmp_fs'


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


def split_captured_command(output: str) -> tuple[str, str]:
    first_line, separator, rest = output.partition('\n')
    if separator and first_line.startswith('[CMD] '):
        return first_line.removeprefix('[CMD] '), rest
    return '(missing command)', output


def capture_frog(*args: str | Path) -> CommandResult:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exit_code = run_frog('py', '-m', 'frog', *args)

    command, body = split_captured_command(buf.getvalue())
    return CommandResult(command, body, exit_code)


def render_result(title: str, result: CommandResult) -> str:
    return f'### {title}\n\nCommand:\n\n{output_fence(result.command)}\nOutput:\n\n{output_fence(result.body)}\n'


def render_run_build(run_result: CommandResult, build_result: CommandResult) -> str:
    if run_result.body == build_result.body:
        commands = f'{run_result.command}\n{build_result.command}\n'
        return (
            f'### Run and Build\n\nCommands:\n\n{output_fence(commands)}\nOutput:\n\n{output_fence(run_result.body)}\n'
        )

    return f'{render_result("Run", run_result)}{render_result("Build", build_result)}'


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
        if run_result.exit_code == 0:
            build_result = capture_frog('-l', 'WARN', 'build', '-r', relative_file)
            output = render_run_build(run_result, build_result)
        else:
            output = render_result('Run', run_result)

        snapshot_name = relative_file.with_suffix('').as_posix()
        write_snapshot(
            dir_snapshots / 'examples' / relative_file.with_suffix('.out').name,
            f'{snapshot_header(snapshot_name)}{render_source(relative_file.as_posix(), file_example.read_text())}{output}',
        )

    for cli_group in cli_example_groups:
        parts = [snapshot_header(f'cli/{cli_group.name}')]
        for cli_example in cli_group.examples:
            print(f'[CLI:{cli_group.name}] {cli_example}')
            args = shlex.split(cli_example)
            result = capture_frog(*args)
            case_name = cli_example if cli_example else '(no arguments)'
            parts.append(f'## Case: `{case_name}`\n\n')
            parts.append(render_cli_sources(args))
            parts.append(render_result('Result', result))

        write_snapshot(dir_snapshots / 'cli' / f'{cli_group.name}.out', ''.join(parts))

    for code_group in code_example_groups:
        parts = [snapshot_header(f'code/{code_group.name}')]
        for index, code_example in enumerate(code_group.examples, start=1):
            print(f'[CODE:{code_group.name}] {index}: {code_example!r}')
            tmp = (tmp_fs / 'code.frog').relative_to(Path.cwd())
            _ = tmp.write_text(code_example)

            run_result = capture_frog('run', tmp)
            build_result = capture_frog('-l', 'WARN', 'build', '-r', tmp)

            parts.append(f'## Case {index:02d}\n\n')
            parts.append(render_source(tmp.as_posix(), code_example))
            parts.append(render_run_build(run_result, build_result))

        write_snapshot(dir_snapshots / 'code' / f'{code_group.name}.out', ''.join(parts))

    for file_code_example in file_code_examples:
        print(f'[FILES] {file_code_example.name}')
        tmp_fs_case = tmp_fs / file_code_example.name
        tmp_fs_case.mkdir(parents=True)
        for file_name, text in file_code_example.files.items():
            file = tmp_fs_case / file_name
            file.parent.mkdir(parents=True, exist_ok=True)
            _ = file.write_text(text)

        main = (tmp_fs_case / file_code_example.main).relative_to(Path.cwd())
        run_result = capture_frog('run', main)
        build_result = capture_frog('-l', 'WARN', 'build', '-r', main)

        parts = [snapshot_header(f'imports/{file_code_example.name}')]
        for file_name, text in file_code_example.files.items():
            parts.append(render_source((tmp_fs_case / file_name).relative_to(Path.cwd()).as_posix(), text))

        parts.append(f'### Main: `{main.as_posix()}`\n\n')
        parts.append(render_run_build(run_result, build_result))
        write_snapshot(dir_snapshots / 'imports' / f'{file_code_example.name}.out', ''.join(parts))
finally:
    shutil.rmtree(tmp_fs, ignore_errors=True)
