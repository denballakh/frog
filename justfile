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
test: check bootstrap-check
    python -m test
    git diff --exit-code HEAD -- test/snapshots
    git status --short -- test/snapshots
    test -z "$(git status --porcelain -- test/snapshots)"

[group("bootstrap")]
frogc-seed:
    mkdir -p build
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 compiler/frogc.c -o build/frogc

[group("bootstrap")]
bootstrap-check: frogc-seed
    build/frogc < compiler/frogc.frog > build/frogc.stage2.c
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 build/frogc.stage2.c -o build/frogc.stage2
    build/frogc.stage2 < compiler/frogc.frog > build/frogc.stage3.c
    cmp compiler/frogc.c build/frogc.stage2.c
    cmp build/frogc.stage2.c build/frogc.stage3.c
    build/frogc < test/bootstrap/read_file.frog > build/frogc.read_file.c
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 build/frogc.read_file.c -o build/frogc.read_file
    build/frogc.read_file > build/frogc.read_file.out
    printf '#\ntrue\n0\nfalse\n' | cmp - build/frogc.read_file.out

[group("bootstrap")]
bootstrap-update: frogc-seed
    build/frogc < compiler/frogc.frog > build/frogc.candidate1.c
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 build/frogc.candidate1.c -o build/frogc.candidate1
    build/frogc.candidate1 < compiler/frogc.frog > build/frogc.candidate2.c
    gcc -std=c11 -pedantic -Wall -Wextra -Wconversion -Werror -O2 build/frogc.candidate2.c -o build/frogc.candidate2
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
