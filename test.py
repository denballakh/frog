import os
from pathlib import Path
import sys

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
    #
    '25 7 + print',
    '25 7 - print',
    '25 7 * print',
    '25 7 / print',
    '25 7 % print',
    '25 7 divmod ? print print',
    '25 7 << print',
    '25 7 >> print',
    '25 7 | print',
    '25 7 & print',
    '25 7 ^ print',
    '25 ~ print',
    #
    'true false and print',
    'true false or print',
    'true not print',
    'false not print',
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
]

tmp = ROOT / 'tmp.lang'
tmp = tmp.relative_to(Path.cwd())
src = ROOT / 'lang.py'
src = src.relative_to(Path.cwd())
for code in code_examples:
    tmp.write_text(code)
    print(f'[CODE] {code!r}')
    res = run(sys.orig_argv[0], src, 'run', tmp)
    print(f'[EXIT CODE] {res}')
    print()

tmp.unlink()
