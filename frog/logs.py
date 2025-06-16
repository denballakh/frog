from typing import Never, Any, cast, assert_never
from pathlib import Path
import sys
import types
from dataclasses import dataclass

from .types import Instruction, Token, Loc, StackEntry, loc_unknown, pp

RC_OK = 0
RC_ERROR = 1
RC_USAGE = 2
RC_INTERNAL_ERROR = 3


LL_ERROR = 0
LL_WARN = 1
LL_INFO = 2
LL_TRACE = 3

LL_DEFAULT = LL_INFO

LL = {
    'ERROR': LL_ERROR,
    'WARN': LL_WARN,
    'INFO': LL_INFO,
    'TRACE': LL_TRACE,
}

LL_FROM_INT = {v: k for k, v in LL.items()}


@dataclass
class _LoggingConfig:
    log_level: int
    log_locations: bool


logging_cfg = _LoggingConfig(LL_DEFAULT, False)

type _Locatable = Instruction | Token | Loc | StackEntry | None


def _locatable_to_loc(loc: _Locatable) -> Loc:
    match loc:
        case Loc():
            return loc
        case Token():
            return loc.loc
        case Instruction():
            return loc.tok.loc
        case StackEntry():
            return loc.tok.loc
        case None:
            return loc_unknown
        case _:
            assert_never(loc)


def note(**notes: Any) -> None:
    for k, v in notes.items():
        print(f'[NOTE] {k}:', end='')
        match v:
            case list():
                v = cast(list[Any], v)
                if not v:
                    print(f' (empty)')

                else:
                    print()
                    for i, x in enumerate(v):
                        print(f'    {i:3}. {pp(x)}')

            case _:
                if not isinstance(v, str):
                    v = pp(v)
                if '\n' not in v:
                    print(f' {v}')

                else:
                    print()
                    for line in v.splitlines():
                        print(f'    {line}')


def get_caller_loc() -> Loc:
    ignored_funcs = {
        get_caller_loc,
        _log,
        error,
        warn,
        info,
        trace,
        typecheck_has_a_bug,
        unreachable,
        notimplemented,
    }
    ignored_func_names = {func.__name__ for func in ignored_funcs}

    frame: types.FrameType | None = sys._getframe(0)  # pyright: ignore[reportPrivateUsage]
    while frame and frame.f_code.co_name in ignored_func_names:
        frame = frame.f_back

    if frame is None:
        print(f'[FATAL] wtf is that')
        sys.exit(RC_INTERNAL_ERROR)

    file = Path(frame.f_code.co_filename)
    root = Path(__file__).parent

    if not file.is_relative_to(root):
        filename = str(file)
    else:
        filename = str(file.relative_to(root))

    return Loc(filename, frame.f_lineno, 0)


def _log(
    level: int,
    loc: _Locatable,
    msg: str,
    orig_loc: Loc | bool | None = None,
    **notes: Any,
) -> None:
    if logging_cfg.log_level >= level:
        print(f'[{LL_FROM_INT[level]}] {_locatable_to_loc(loc)}: {msg}')
        if orig_loc is None and logging_cfg.log_locations or orig_loc is True:
            orig_loc = get_caller_loc()
        if orig_loc is not None:
            print(f'[LOC] {orig_loc}')
        note(**notes)


def error(loc: _Locatable, msg: str, exitcode: int = RC_ERROR, **notes: Any) -> Never:
    _log(LL_ERROR, loc, msg, **notes)
    sys.exit(exitcode)


def warn(loc: _Locatable, msg: str, **notes: Any) -> None:
    _log(LL_WARN, loc, msg, **notes)


def info(loc: _Locatable, msg: str, **notes: Any) -> None:
    _log(LL_INFO, loc, msg, **notes)


def trace(loc: _Locatable, msg: str, **notes: Any) -> None:
    _log(LL_TRACE, loc, msg, **notes)


def typecheck_has_a_bug(loc: _Locatable, msg: str, **notes: Any) -> Never:
    msg = f'typecheck has a bug: {msg}'
    _log(LL_ERROR, loc, msg, orig_loc=True, **notes)
    sys.exit(RC_INTERNAL_ERROR)


def unreachable(loc: _Locatable, msg: str = '<?>', **notes: Any) -> Never:
    msg = f'unreachable: {msg}'
    _log(LL_ERROR, loc, msg, orig_loc=True, **notes)
    sys.exit(RC_INTERNAL_ERROR)


def notimplemented(loc: _Locatable, msg: str) -> Never:
    msg = f'not implemented: {msg}'
    _log(LL_ERROR, loc, msg, orig_loc=True)
    sys.exit(RC_INTERNAL_ERROR)
