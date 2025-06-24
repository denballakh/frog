import contextlib
from pathlib import Path
import io
import shlex

from frog.__main__ import run_frog


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
    #
    '5 int ? cast ? print',
    '5 bool cast print',
    '0 bool cast print',
    'true int cast print',
    'false int cast print',
    'false bool cast print',
]
cli_examples = [
    '',
    '-h',
    '--help',
    'run',
    'run xxx',
    '-l',
    '-l TRACE',
    '-l TRACE run',
    '-l TRACE run examples/01_simple.frog',
    '-l LOL run examples/01_simple.frog',
    '-l WARN run examples/01_simple.frog',
    '-l INFO run examples/01_simple.frog',
    '-l ERROR run examples/01_simple.frog',
    #
    '-l TRACE run examples/02_while.frog',
    '-l TRACE build -r examples/02_while.frog',
]

ROOT = Path(__file__).parent.parent

dir_examples = ROOT / 'examples'
dir_tests = ROOT / 'test'

for file_example in sorted(dir_examples.iterdir()):
    if not file_example.is_file():
        continue
    if file_example.suffix != '.frog':
        continue
    file_example = file_example.relative_to(ROOT)

    print(f'[FILE] {file_example}')
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = run_frog('py', '-m', 'frog', 'run', file_example)
            if res == 0:
                res = run_frog('py', '-m', 'frog', '-l', 'WARN', 'build', '-r', file_example)
    finally:
        _ = (dir_tests / file_example.name).with_suffix('.out').write_text(buf.getvalue())


buf = io.StringIO()
try:
    for cli_example in cli_examples:
        print(f'[CLI] {cli_example}')
        with contextlib.redirect_stdout(buf):
            res = run_frog('py', '-m', 'frog', *shlex.split(cli_example))
finally:
    _ = (dir_tests / 'cli.out').write_text(buf.getvalue())


tmp = dir_tests / 'tmp.frog'
tmp = tmp.relative_to(Path.cwd())

buf = io.StringIO()
try:
    for code_example in code_examples:
        print(f'[CODE] {code_example!r}')
        with contextlib.redirect_stdout(buf):
            _ = tmp.write_text(code_example)
            res = run_frog('py', '-m', 'frog', 'run', tmp)
            res = run_frog('py', '-m', 'frog', '-l', 'WARN', 'build', '-r', tmp)
finally:
    _ = (dir_tests / 'code.out').write_text(buf.getvalue())
tmp.unlink()
