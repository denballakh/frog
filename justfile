# https://just.systems/man/en/

@_default:
    just --list

@_mypy:
    mypy .

@_basedpyright:
    basedpyright .

@_black:
    black .

@_vscode-grammar:
    python test/vscode_grammar.py

[group("test")]
typecheck: _mypy _basedpyright
[group("test")]
fmt: _black
[group("test")]
check: typecheck fmt _vscode-grammar

[group("test")]
test: check bootstrap-check
    python -m test
    git diff --exit-code HEAD -- test/snapshots
    git status --short -- test/snapshots
    test -z "$(git status --porcelain -- test/snapshots)"

_compile-frogc source output:
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 "{{source}}" -o "{{output}}"

_compile-frogc-filter source output:
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 "{{source}}" -o "{{output}}"

[group("bootstrap")]
frogc-seed:
    mkdir -p build
    just _compile-frogc compiler/frogc.c build/frogc

[group("bootstrap")]
bootstrap-check: frogc-seed
    build/frogc < compiler/frogc.frog > build/frogc.stage2.c
    just _compile-frogc build/frogc.stage2.c build/frogc.stage2
    build/frogc.stage2 < compiler/frogc.frog > build/frogc.stage3.c
    just _compile-frogc build/frogc.stage3.c build/frogc.stage3
    cmp compiler/frogc.c build/frogc.stage2.c
    cmp build/frogc.stage2.c build/frogc.stage3.c

[group("bootstrap")]
bootstrap-update: frogc-seed
    build/frogc < compiler/frogc.frog > build/frogc.candidate1.c
    just _compile-frogc-filter build/frogc.candidate1.c build/frogc.candidate1
    build/frogc.candidate1 < compiler/frogc.frog > build/frogc.candidate2.c
    just _compile-frogc-filter build/frogc.candidate2.c build/frogc.candidate2
    build/frogc.candidate2 < compiler/frogc.frog > build/frogc.candidate3.c
    cmp build/frogc.candidate2.c build/frogc.candidate3.c
    cp build/frogc.candidate2.c compiler/frogc.c

[group("test")]
show-diff:
    git diff -- test/snapshots
    git status --short -- test/snapshots

# ONLY run this if you are ABSOLUTELY SURE the snapshot output changes are correct.
[group("test")]
approve-diff:
    git add -A test/snapshots

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
