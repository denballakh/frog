
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

repl:
    ./lang.py repl

test: && check
    ./test.py

clean:
    rm *.c || true
    rm *.exe || true
    rm examples/*.c || true
    rm examples/*.exe || true
