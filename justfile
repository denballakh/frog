
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
check: typecheck fmt

repl:
    python -m frog repl

test:
    python -m test

clean:
    rm *.c || true
    rm *.exe || true
    rm examples/*.c || true
    rm examples/*.exe || true

precommit: check test clean
