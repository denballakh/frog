# https://just.systems/man/en/

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
test: check
    python -m test
    git diff --exit-code HEAD -- test/snapshots
    git status --short -- test/snapshots
    test -z "$(git status --porcelain -- test/snapshots)"

[group("test")]
show-diff:
    git diff -- test/snapshots
    git status --short -- test/snapshots

# ONLY run this if you are ABSOLUTELY SURE the snapshot output changes are correct.
[group("test")]
approve-diff:
    git add -A test/snapshots

[group("run")]
@repl:
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
