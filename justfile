
@_default:
    just --list

@_mypy:
    mypy .

@_basedpyright:
    basedpyright .

@_black:
    black .

typecheck: _mypy _basedpyright
fmt: _black
check: typecheck && fmt

run file: check
    python lang.py run {{file}}

repl: check
    python lang.py repl

test: && check
    python test.py

