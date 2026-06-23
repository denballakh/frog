
@_default:
    just --list

@_mypy:
    mypy .

@_basedpyright:
    basedpyright .

@_black:
    black .

[group("test")]
typecheck: _mypy _basedpyright
[group("test")]
fmt: _black
[group("test")]
check: typecheck fmt

[group("test")]
test: && check
    rm test/*.out || true
    python -m test

[group("run")]
repl:
    python -m frog repl

[group("run")]
[positional-arguments]
@cli *args:
    python -m frog "$@"

[group("misc")]
clean:
    rm *.c || true
    rm *.exe || true
    rm examples/*.c || true
    rm examples/*.exe || true
    rm test/*.c || true
    rm test/*.exe || true
