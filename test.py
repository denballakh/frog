import contextlib
import os
from pathlib import Path
import sys
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tqdm import tqdm
else:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm[T](x: T, *_: Any, **__: Any) -> T:
            return x

ROOT = Path(__file__).parent


def run(*cmds: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> int:
    import subprocess

    cmd = subprocess.list2cmdline(cmds)
    print(f'[CMD] {cmd}')
    res = subprocess.run(
        cmds,
        shell=True,
        capture_output=True,
        universal_newlines=True,
    )
    out = res.stdout
    err = res.stderr
    if out:
        print(f'[STDOUT]:')
        print(out)
    if err:
        print(f'[STDERR]:')
        print(err)
    return res.returncode


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
]

src = ROOT / 'lang.py'
src = src.relative_to(Path.cwd())

tmp = ROOT / 'tmp.lang'
tmp = tmp.relative_to(Path.cwd())

out = ROOT / 'test_dump.txt'
buf = io.StringIO()
for code in tqdm(code_examples):
    with contextlib.redirect_stdout(buf):
        _ = tmp.write_text(code)
        print(f'[CODE] {code!r}')
        res = run(sys.orig_argv[0], src, 'run', tmp)
        print(f'[EXIT CODE] {res}')
        print()
_ = out.write_text(buf.getvalue())
tmp.unlink()
