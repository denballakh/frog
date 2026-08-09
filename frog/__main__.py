from collections.abc import Generator
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import IO

GCC_FLAGS = ('-std=c11', '-pedantic', '-Wall', '-Wextra', '-Wconversion', '-Werror', '-O2')
COMPILER_PREPARE_ATTEMPTS = 3
USAGE = """Usage: python -m frog <command> [options]

Commands:
  run [-c CODE | FILE]       compile and run Frog source
  build [-o FILE] [-r] FILE  compile Frog source to a binary
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def usage_error(message: str) -> int:
    print(f'frog: {message}', file=sys.stderr)
    print('Try `python -m frog --help`.', file=sys.stderr)
    return 2


def release_locks(locks: list[IO[bytes]]) -> None:
    for lock in reversed(locks):
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            print(f'frog: unable to release output lock: {error}', file=sys.stderr)
        finally:
            try:
                lock.close()
            except OSError as error:
                print(f'frog: unable to close output lock: {error}', file=sys.stderr)


@contextmanager
def output_locks(destinations: tuple[Path, ...]) -> Generator[bool]:
    lock_directory = repo_root() / 'build' / 'locks'
    locks: list[IO[bytes]] = []
    try:
        lock_directory.mkdir(parents=True, exist_ok=True)
        lock_paths: set[Path] = set()
        for destination in destinations:
            absolute = destination.absolute()
            identity = absolute.parent.resolve() / absolute.name
            digest = hashlib.sha256(os.fsencode(identity)).hexdigest()
            lock_paths.add(lock_directory / f'{digest}.lock')
        for lock_path in sorted(lock_paths):
            lock = lock_path.open('a+b')
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except OSError:
                lock.close()
                raise
            locks.append(lock)
    except OSError as error:
        release_locks(locks)
        print(f'frog: unable to acquire output lock: {error}', file=sys.stderr)
        yield False
        return

    try:
        yield True
    finally:
        release_locks(locks)


def compiler_paths() -> tuple[Path, Path, Path]:
    root = repo_root()
    return root / 'compiler' / 'frogc.c', root / 'build' / 'frogc', root / 'build' / 'frogc.sha256'


def compiler_path() -> Path | None:
    source, compiler, stamp = compiler_paths()
    with output_locks((compiler, stamp)) as locked:
        if not locked:
            return None
        for _ in range(COMPILER_PREPARE_ATTEMPTS):
            try:
                source_bytes = source.read_bytes()
            except FileNotFoundError:
                print(f'frog: checked compiler source not found: {source}', file=sys.stderr)
                return None
            except OSError as error:
                print(f'frog: unable to read checked compiler source {source}: {error}', file=sys.stderr)
                return None

            digest = hashlib.sha256(source_bytes)
            for flag in GCC_FLAGS:
                digest.update(b'\0')
                digest.update(flag.encode('ascii'))
            source_stamp = digest.hexdigest().encode('ascii')

            try:
                if compiler.is_file() and os.access(compiler, os.X_OK) and stamp.read_bytes() == source_stamp:
                    return compiler.resolve()
            except OSError:
                pass

            try:
                compiler.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='frogc-', dir=compiler.parent) as directory:
                    candidate = Path(directory) / 'frogc'
                    stamp_candidate = Path(directory) / 'frogc.sha256'
                    result = subprocess.run(
                        ['gcc', *GCC_FLAGS, '-x', 'c', '-', '-o', str(candidate)],
                        input=source_bytes,
                    )
                    if result.returncode != 0:
                        return None
                    if source.read_bytes() != source_bytes:
                        continue
                    _ = stamp_candidate.write_bytes(source_stamp)
                    os.replace(candidate, compiler)
                    os.replace(stamp_candidate, stamp)
                    if source.read_bytes() == source_bytes:
                        return compiler.resolve()
            except OSError as error:
                print(f'frog: unable to prepare native compiler: {error}', file=sys.stderr)
                return None
        print('frog: checked compiler source kept changing while preparing the native compiler', file=sys.stderr)
        return None


def temporary_sibling(destination: Path) -> Path | None:
    try:
        descriptor, name = tempfile.mkstemp(prefix=f'.{destination.name}.', dir=destination.parent)
    except OSError as error:
        print(f'frog: unable to create output beside {destination}: {error}', file=sys.stderr)
        return None
    os.close(descriptor)
    return Path(name)


def generate_c(compiler: Path, source: bytes, source_directory: Path, generated_c: Path) -> int:
    candidate = temporary_sibling(generated_c)
    if candidate is None:
        return 1
    try:
        with candidate.open('wb') as output:
            result = subprocess.run([str(compiler)], input=source, stdout=output, cwd=source_directory)
    except OSError as error:
        print(f'frog: unable to run compiler: {error}', file=sys.stderr)
        candidate.unlink(missing_ok=True)
        return 1
    if result.returncode != 0:
        candidate.unlink(missing_ok=True)
        return 1
    try:
        os.replace(candidate, generated_c)
    except OSError as error:
        print(f'frog: unable to write generated C to {generated_c}: {error}', file=sys.stderr)
        candidate.unlink(missing_ok=True)
        return 1
    return 0


def compile_c(generated_c: Path, executable: Path) -> int:
    try:
        result = subprocess.run(['gcc', *GCC_FLAGS, '-x', 'c', str(generated_c), '-o', str(executable)])
    except OSError as error:
        print(f'frog: unable to run gcc: {error}', file=sys.stderr)
        return 1
    return result.returncode


def destination_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def publish_build(
    generated_c_candidate: Path,
    generated_c: Path,
    executable_candidate: Path,
    executable: Path,
) -> int:
    artifacts = (
        (generated_c_candidate, generated_c),
        (executable_candidate, executable),
    )
    for _, destination in artifacts:
        if destination.is_dir():
            print(f'frog: output path is a directory: {destination}', file=sys.stderr)
            return 1

    backups: list[tuple[Path, Path]] = []
    for _, destination in artifacts:
        if not destination_exists(destination):
            continue
        backup = temporary_sibling(destination)
        if backup is None:
            for existing_backup, _ in backups:
                existing_backup.unlink(missing_ok=True)
            return 1
        try:
            backup.unlink()
        except OSError as error:
            print(f'frog: unable to prepare build-artifact backup {backup}: {error}', file=sys.stderr)
            for existing_backup, _ in backups:
                existing_backup.unlink(missing_ok=True)
            return 1
        backups.append((backup, destination))

    moved_backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for backup, destination in backups:
            os.replace(destination, backup)
            moved_backups.append((backup, destination))
        for candidate, destination in artifacts:
            os.replace(candidate, destination)
            published.append(destination)
    except OSError as error:
        print(f'frog: unable to publish build artifacts: {error}', file=sys.stderr)
        rollback_errors: list[OSError] = []
        for destination in reversed(published):
            try:
                destination.unlink(missing_ok=True)
            except OSError as cleanup_error:
                rollback_errors.append(cleanup_error)
        for backup, destination in reversed(moved_backups):
            try:
                os.replace(backup, destination)
            except OSError as restore_error:
                rollback_errors.append(restore_error)
        for reported_error in rollback_errors:
            print(f'frog: unable to restore a prior build artifact: {reported_error}', file=sys.stderr)
        return 1

    for backup, _ in moved_backups:
        try:
            backup.unlink()
        except OSError as error:
            print(f'frog: warning: unable to remove build-artifact backup {backup}: {error}', file=sys.stderr)
    return 0


def build_source(
    compiler: Path,
    source: bytes,
    source_directory: Path,
    generated_c: Path,
    executable: Path,
) -> int:
    generated_c_candidate = temporary_sibling(generated_c)
    if generated_c_candidate is None:
        return 1
    executable_candidate: Path | None = None
    try:
        if generate_c(compiler, source, source_directory, generated_c_candidate) != 0:
            return 1
        executable_candidate = temporary_sibling(executable)
        if executable_candidate is None:
            return 1
        if compile_c(generated_c_candidate, executable_candidate) != 0:
            return 1
        return publish_build(generated_c_candidate, generated_c, executable_candidate, executable)
    finally:
        generated_c_candidate.unlink(missing_ok=True)
        if executable_candidate is not None:
            executable_candidate.unlink(missing_ok=True)


def run_executable(executable: Path) -> int:
    try:
        result = subprocess.run([str(executable.resolve())])
    except OSError as error:
        print(f'frog: unable to run {executable}: {error}', file=sys.stderr)
        return 1
    if result.returncode < 0:
        return 128 - result.returncode
    return result.returncode


def run_source(source: bytes, source_directory: Path) -> int:
    compiler = compiler_path()
    if compiler is None:
        return 1
    with tempfile.TemporaryDirectory(prefix='frog-run-') as directory:
        temporary = Path(directory)
        generated_c = temporary / 'program.c'
        executable = temporary / 'program'
        if generate_c(compiler, source, source_directory, generated_c) != 0:
            return 1
        if compile_c(generated_c, executable) != 0:
            return 1
        return run_executable(executable)


def read_source(file: Path) -> bytes | None:
    if not file.is_file():
        print(f'frog: source file not found: {file}', file=sys.stderr)
        return None
    try:
        return file.read_bytes()
    except OSError as error:
        print(f'frog: unable to read {file}: {error}', file=sys.stderr)
        return None


def paths_alias(first: Path, second: Path) -> bool:
    try:
        if first.exists() and second.exists() and os.path.samefile(first, second):
            return True
        return first.resolve() == second.resolve()
    except OSError:
        return first.absolute() == second.absolute()


def run_command(argv: list[str]) -> int:
    if argv == ['-h'] or argv == ['--help']:
        print('Usage: python -m frog run [-c CODE | FILE]')
        return 0
    if not argv:
        return usage_error('run requires a source file or -c CODE')
    if argv[0] == '-c':
        if len(argv) != 2:
            return usage_error('run -c requires exactly one CODE argument')
        return run_source(argv[1].encode(), Path.cwd())
    if argv[0].startswith('-'):
        return usage_error(f'unknown run option: {argv[0]}')
    if len(argv) != 1:
        return usage_error('run accepts exactly one source file')
    source_file = Path(argv[0])
    source = read_source(source_file)
    if source is None:
        return 1
    return run_source(source, source_file.absolute().parent)


def build_command(argv: list[str]) -> int:
    output: Path | None = None
    should_run = False
    while argv and argv[0].startswith('-'):
        option = argv.pop(0)
        if option in {'-h', '--help'}:
            print('Usage: python -m frog build [-o FILE] [-r] FILE')
            return 0
        if option == '-r':
            should_run = True
            continue
        if option == '-o':
            if not argv:
                return usage_error('build -o requires an output file')
            output = Path(argv.pop(0))
            continue
        return usage_error(f'unknown build option: {option}')

    if len(argv) != 1:
        return usage_error('build requires exactly one source file')
    source_file = Path(argv[0])
    source = read_source(source_file)
    if source is None:
        return 1

    generated_c = source_file.with_suffix('.c')
    executable = output if output is not None else source_file.with_suffix('.exe')
    if paths_alias(source_file, generated_c):
        return usage_error('generated C path aliases the source file')
    if paths_alias(source_file, executable):
        return usage_error('executable path aliases the source file')
    if paths_alias(generated_c, executable):
        return usage_error('executable path aliases the generated C file')
    _, cached_compiler, compiler_stamp = compiler_paths()
    for artifact in (generated_c, executable):
        if paths_alias(artifact, cached_compiler) or paths_alias(artifact, compiler_stamp):
            return usage_error('build output aliases the native compiler cache')

    compiler = compiler_path()
    if compiler is None:
        return 1
    with output_locks((generated_c, executable)) as locked:
        if not locked:
            return 1
        if build_source(compiler, source, source_file.absolute().parent, generated_c, executable) != 0:
            return 1
        if should_run:
            return run_executable(executable)
        return 0


def main(argv: list[str]) -> int:
    if not argv:
        return usage_error('missing command')
    if argv[0] in {'-h', '--help'}:
        print(USAGE, end='')
        return 0

    command, *arguments = argv
    if command == 'run':
        return run_command(arguments)
    if command == 'build':
        return build_command(arguments)
    return usage_error(f'unknown command: {command}')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
