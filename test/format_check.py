import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

ROOT = Path(__file__).parent.parent
FROGC = ROOT / 'build' / 'frogc'
EXCLUSIONS = ROOT / 'test' / 'formatter_exclusions.tsv'
FORMATTER_TIMEOUT_SECONDS = 120
EXCLUSION_KINDS = frozenset({'invalid'})


def tracked_frog_paths() -> list[str]:
    result = subprocess.run(
        ['git', 'ls-files', '-z', '--', '*.frog'],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = os.fsdecode(result.stderr).rstrip()
        raise RuntimeError(f'git ls-files failed with status {result.returncode}: {diagnostic}')
    if result.stdout and not result.stdout.endswith(b'\0'):
        raise RuntimeError('git ls-files returned a non-NUL-terminated path list')
    return [os.fsdecode(path) for path in result.stdout.split(b'\0') if path]


def load_exclusions() -> tuple[dict[str, str], list[str]]:
    exclusions: dict[str, str] = {}
    paths: list[str] = []
    errors: list[str] = []
    for line_number, line in enumerate(EXCLUSIONS.read_text(encoding='utf-8').splitlines(), start=1):
        if not line or line.startswith('#'):
            continue
        fields = line.split('\t')
        if len(fields) != 2:
            errors.append(f'{EXCLUSIONS.relative_to(ROOT)}:{line_number}: expected KIND<TAB>PATH')
            continue
        kind, path = fields
        if kind not in EXCLUSION_KINDS:
            errors.append(f'{EXCLUSIONS.relative_to(ROOT)}:{line_number}: unknown exclusion kind: {kind}')
        if not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
            errors.append(f'{EXCLUSIONS.relative_to(ROOT)}:{line_number}: invalid repository-relative path: {path}')
        if path in exclusions:
            errors.append(f'{EXCLUSIONS.relative_to(ROOT)}:{line_number}: duplicate path: {path}')
            continue
        exclusions[path] = kind
        paths.append(path)
    if paths != sorted(paths):
        errors.append(f'{EXCLUSIONS.relative_to(ROOT)}: paths must be sorted')
    return exclusions, errors


def format_path(path: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [FROGC, 'fmt', path],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=FORMATTER_TIMEOUT_SECONDS,
        check=False,
    )


def formatter_diagnostic(result: subprocess.CompletedProcess[bytes]) -> str:
    diagnostic = result.stderr.decode('utf-8', errors='backslashreplace').rstrip()
    return diagnostic or '<empty stderr>'


def check_path(path: str, exclusion_kind: str | None) -> list[str]:
    command = shlex.join([str(FROGC), 'fmt', path])
    source_path = ROOT / path
    try:
        source = source_path.read_bytes()
        result = format_path(path)
    except OSError as error:
        return [f'{path}: cannot check formatter: {error}']
    except subprocess.TimeoutExpired:
        return [f'{path}: {command} exceeded {FORMATTER_TIMEOUT_SECONDS} seconds']

    errors: list[str] = []
    try:
        source_after_format = source_path.read_bytes()
    except OSError as error:
        errors.append(f'{path}: cannot read source after formatter invocation: {error}')
    else:
        if source_after_format != source:
            errors.append(f'{path}: formatter modified its input file')

    if exclusion_kind is not None:
        if result.returncode == 0:
            errors.append(f'{path}: stale {exclusion_kind} exclusion; formatter now succeeds')
        elif result.returncode != 1:
            errors.append(f'{path}: excluded formatter invocation returned status {result.returncode}, expected 1')
        if result.stdout:
            errors.append(f'{path}: excluded formatter failure wrote {len(result.stdout)} bytes to stdout')
        if not result.stderr:
            errors.append(f'{path}: excluded formatter failure produced no diagnostic')
        return errors

    if result.returncode != 0:
        errors.append(
            f'{path}: unclassified formatter failure with status {result.returncode}:\n'
            + f'{formatter_diagnostic(result)}'
        )
        return errors
    if result.stderr:
        errors.append(f'{path}: successful formatter invocation wrote to stderr:\n{formatter_diagnostic(result)}')
    if result.stdout != source:
        errors.append(f'{path}: formatter output differs; run {command} and replace the file')
    return errors


def main() -> int:
    try:
        tracked_paths = tracked_frog_paths()
        exclusions, errors = load_exclusions()
    except (OSError, RuntimeError) as setup_error:
        print(f'format check setup failed: {setup_error}', file=sys.stderr)
        return 1

    tracked_set = set(tracked_paths)
    for path in exclusions.keys() - tracked_set:
        errors.append(f'{EXCLUSIONS.relative_to(ROOT)}: exclusion is not a tracked Frog file: {path}')
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1

    for path in tracked_paths:
        errors.extend(check_path(path, exclusions.get(path)))

    if errors:
        print('Frog format check failed:', file=sys.stderr)
        for message in errors:
            print(f'- {message}', file=sys.stderr)
        return 1

    invalid_count = sum(kind == 'invalid' for kind in exclusions.values())
    print(
        f'checked {len(tracked_paths)} tracked Frog files: '
        + f'{len(tracked_paths) - len(exclusions)} formatted, '
        + f'{invalid_count} expected invalid'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
