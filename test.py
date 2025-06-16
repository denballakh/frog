#!/usr/bin/env python3
import contextlib
from pathlib import Path
import io
from typing import TYPE_CHECKING, Any
import shlex

import lang

if TYPE_CHECKING:
    from tqdm import tqdm
else:
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm[T](x: T, *_: Any, **__: Any) -> T:
            return x


ROOT = Path(__file__).parent

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
    '-l TRACE run examples/01_simple.lang',
    '-l LOL run examples/01_simple.lang',
    '-l WARN run examples/01_simple.lang',
    '-l INFO run examples/01_simple.lang',
    '-l ERROR run examples/01_simple.lang',
    '-l DEFAULT run examples/01_simple.lang',
    #
    '-l TRACE run examples/02_while.lang',
    '-l TRACE build -r examples/02_while.lang',
]

dir_examples = ROOT / 'examples'

for file_example in dir_examples.iterdir():
    file_example = file_example.relative_to(Path.cwd())
    if not file_example.is_file():
        continue
    if file_example.suffix != '.lang':
        continue
    cli_examples.append(f'run {file_example}')
    cli_examples.append(f'build -r {file_example}')

src = ROOT / 'lang.py'
src = src.relative_to(Path.cwd())

tmp = ROOT / 'tmp.lang'
tmp = tmp.relative_to(Path.cwd())

out = ROOT / 'test_dump.txt'
buf = io.StringIO()
try:
    for code in tqdm(code_examples):
        with contextlib.redirect_stdout(buf):
            _ = tmp.write_text(code)
            print('=' * 60)
            print(f'[CODE] {code!r}')
            # res = run_cmd('py', src, 'run', tmp)
            res = lang.run_lang('py', src, 'run', tmp)
            print(f'[EXIT CODE] {res}')
            if res == 0:
                res = lang.run_lang('py', src, '-l', 'WARN', 'build', '-r', tmp)
                print(f'[EXIT CODE] {res}')
            print()

    for cli in tqdm(cli_examples):
        with contextlib.redirect_stdout(buf):
            print('=' * 60)
            res = lang.run_cmd('py', src, *shlex.split(cli))
            print(f'[EXIT CODE] {res}')

except Exception as e:
    print(buf.getvalue())
    import traceback

    traceback.print_exc()

else:
    _ = out.write_text(buf.getvalue())

finally:
    tmp.unlink()
