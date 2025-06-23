import os
import subprocess
from pathlib import Path
import sys
import traceback

from . import tokenize
from . import compile
from . import typecheck
from . import interpret
from . import translate
from .types import Loc
from .logs import error, LL, LL_DEFAULT
from .logs import logging_cfg
from .logs import RC_OK, RC_USAGE, RC_ERROR, RC_INTERNAL_ERROR


def repl() -> None:
    import traceback

    try:
        import readline  # pyright: ignore[reportUnusedImport]
    except ImportError:
        pass

    while True:
        try:
            line = input('> ')
            if line == 'q':
                break

            toks = tokenize(line, filename='<repl>')
            ir = compile(toks)
            typecheck(ir)
            interpret(ir)

        except EOFError:
            break
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc()


def run_cmd(*cmds: str | os.PathLike[str]) -> int:

    cmd = subprocess.list2cmdline(cmds)
    print(f'[CMD] {cmd}')

    res = subprocess.run(
        cmds,
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

    if res.returncode != 0:
        print(f'[EXIT CODE] {res.returncode}')

    return res.returncode


def run_frog(*args: str | os.PathLike[str]) -> int:

    cmd = subprocess.list2cmdline([*args])
    print(f'[CMD] {cmd}')

    try:
        main([str(x) for x in args[3:]])

    except SystemExit as e:
        code = e.code
        assert isinstance(code, int), f'unknown exit code: {code!r}'
    else:
        code = RC_OK

    if code != 0:
        print(f'[EXIT CODE] {code}')

    return code


USAGE = """
Options:
  -h --help                   print this help message
  -l <level>                  log level: ERROR,WARN,INFO,TRACE
Subcommands:
  run [OPTIONS]             interpre
    -c CODE                   code to interpret
       FILE                   file to interpret
  build [OPTIONS] FILE      build
    FILE                      file to build
    OPTIONS:
      -o FILE                 where to put built binary
      -r                      also run the binary
  repl                      start a Read-Eval-Print-Loop

"""


def main(argv: list[str]) -> None:
    def usage_short() -> None:
        print(f'Usage: py -m frog [OPTIONS] SUBCOMMAND <ARGS>')

    def usage() -> None:
        usage_short()
        print(USAGE)

    logging_cfg.log_level = LL_DEFAULT

    while argv:
        if argv[0] == '-h' or argv[0] == '--help':
            usage()
            sys.exit(RC_OK)

        elif argv[0] == '-l':
            _, *argv = argv
            if len(argv) < 1:
                usage_short()
                print(f'[ERROR] no log level specified')
                sys.exit(RC_USAGE)

            ll_str, *argv = argv
            if ll_str not in LL:
                error(Loc('<cli>', 1, 0), f'invalid log level: {ll_str}, expected one of {list(LL)}')
            logging_cfg.log_level = LL[ll_str]

        else:
            break

    if len(argv) < 1:
        usage_short()
        print(f'[ERROR] no subcommand specified')
        sys.exit(RC_USAGE)

    subcmd, *argv = argv

    if subcmd == 'run':
        code_src: str | None = None
        filename: str

        while len(argv) > 0:
            if argv[0] == '-h':
                usage()
                sys.exit(RC_OK)

            elif argv[0] == '-c':
                _, *argv = argv
                if len(argv) < 1:
                    usage_short()
                    print(f'[ERROR] no code specified')
                    sys.exit(RC_USAGE)

                code_src = argv[0]
                argv = argv[1:]

            else:
                break

        if code_src is None:
            if len(argv) < 1:
                usage_short()
                print(f'[ERROR] no file specified')
                sys.exit(RC_USAGE)

            file_src = Path(argv[0])
            argv = argv[1:]

            if not file_src.exists():
                print(f'[ERROR] file {file_src} does not exist')
                sys.exit(RC_ERROR)

            code_src = file_src.read_text()
            filename = str(file_src)

        else:
            filename = '<cli>'

        if len(argv) > 0:
            usage_short()
            print(f'[ERROR] unrecognized arguments: {argv}')
            sys.exit(RC_USAGE)

        toks = tokenize(code_src, filename=filename)
        ir = compile(toks)
        typecheck(ir)

        try:
            interpret(ir)
        except Exception:
            traceback.print_exc()
            sys.exit(RC_INTERNAL_ERROR)

        sys.exit(RC_OK)

    elif subcmd == 'build':
        file_out: Path | None = None
        should_run = False

        while len(argv) > 0:
            if argv[0] == '-h':
                usage()
                sys.exit(RC_OK)

            elif argv[0] == '-o':
                _, *argv = argv
                if len(argv) < 1:
                    usage_short()
                    print(f'[ERROR] no output file specified')
                    sys.exit(RC_USAGE)
                file_out = Path(argv[0])
                argv = argv[1:]

            elif argv[0] == '-r':
                should_run = True
                argv = argv[1:]

            else:
                break

        if len(argv) < 1:
            usage_short()
            print(f'[ERROR] no file specified')
            sys.exit(RC_USAGE)

        file_src = Path(argv[0])
        argv = argv[1:]
        file_c = file_src.with_suffix('.c')

        if file_out is None:
            file_out = file_src.with_suffix('.exe')

        if len(argv) > 0:
            usage_short()
            print(f'[ERROR] unrecognized arguments: {argv}')
            sys.exit(RC_USAGE)

        if not file_src.exists():
            print(f'[ERROR] file {file_src} does not exist')
            sys.exit(RC_ERROR)

        text = file_src.read_text()
        toks = tokenize(text, filename=str(file_src))
        ir = compile(toks)
        typecheck(ir)

        code = translate(ir)
        with open(file_c, 'wt') as f:
            _ = f.write(code)

        ret = run_cmd('gcc', file_c, '-o', file_out)
        if ret != 0:
            error(Loc('<cli>', 1, 0), f'gcc failed with exit code {ret}')

        if should_run:
            ret = run_cmd(f'./{file_out}')
            if ret != 0:
                error(Loc('<cli>', 1, 0), f'{file_out} failed with exit code {ret}')

        sys.exit(RC_OK)

    elif subcmd == 'repl':
        repl()
        sys.exit(RC_OK)

    else:
        usage_short()
        print(f'[ERROR] unknown subcommand: {subcmd}')
        sys.exit(RC_USAGE)


if __name__ == '__main__':
    main(sys.argv[1:])
