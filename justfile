# https://just.systems/man/en/

@_default:
    just --list

@_mypy:
    mypy .

@_basedpyright:
    basedpyright .

@_black:
    black .

@_black-check:
    black --check .

@_vscode-grammar:
    python test/vscode_grammar.py

[group("test")]
typecheck: _mypy _basedpyright
[group("test")]
fmt: _black
[group("test")]
check: typecheck _black-check _vscode-grammar

[group("test")]
test: check bootstrap-check frog-regressions
    python -m test

[group("test")]
frog-regressions: frogc-seed
    build/frogc build -o build/frog-tests test/runner.frog
    python -m test --frog-only

_compile-frogc source output:
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 "{{source}}" -o "{{output}}"

[group("bootstrap")]
frogc-seed:
    mkdir -p build
    just _compile-frogc compiler/frogc.c build/frogc

[group("bootstrap")]
bootstrap-check: frogc-seed
    cd compiler && ../build/frogc < frogc.frog > ../build/frogc.stage2.c
    just _compile-frogc build/frogc.stage2.c build/frogc.stage2
    cd compiler && ../build/frogc.stage2 < frogc.frog > ../build/frogc.stage3.c
    just _compile-frogc build/frogc.stage3.c build/frogc.stage3
    cmp compiler/frogc.c build/frogc.stage2.c
    cmp build/frogc.stage2.c build/frogc.stage3.c

[group("bootstrap")]
bootstrap-update: frogc-seed
    cd compiler && ../build/frogc < frogc.frog > ../build/frogc.candidate1.c
    just _compile-frogc build/frogc.candidate1.c build/frogc.candidate1
    cd compiler && ../build/frogc.candidate1 < frogc.frog > ../build/frogc.candidate2.c
    just _compile-frogc build/frogc.candidate2.c build/frogc.candidate2
    cd compiler && ../build/frogc.candidate2 < frogc.frog > ../build/frogc.candidate3.c
    cmp build/frogc.candidate2.c build/frogc.candidate3.c
    cp build/frogc.candidate2.c compiler/frogc.c

[group("run")]
[positional-arguments]
@cli *args: frogc-seed
    build/frogc "$@"

[group("misc")]
[confirm('Install vscode extension? [y/N]')]
vscode-install:
    mkdir -p "$HOME/.vscode/extensions"
    ln -sfn "{{justfile_directory()}}/ide/vscode" "$HOME/.vscode/extensions/frog"

[group("misc")]
clean:
    rm *.c || true
    rm *.exe || true
    rm examples/*.c || true
    rm examples/*.exe || true
    rm test/*.c || true
    rm test/*.exe || true
