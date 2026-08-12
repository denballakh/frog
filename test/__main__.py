from collections.abc import Mapping
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).parent.parent
FROGC = ROOT / 'build' / 'frogc'
FROG_TESTS = ROOT / 'build' / 'frog-tests'
FROG_TEST_EXECUTABLE = ROOT / 'build' / 'frog-test-case.exe'
HOST_POLICY_FIXTURES = ROOT / 'test' / 'cases' / 'host_policy'
TMP_FS = ROOT / 'test' / 'tmp_fs'
CLI_BUILD_ARTIFACTS = [ROOT / 'test' / 'cases' / 'cli' / f'build_run.{suffix}' for suffix in ('c', 'exe')]
COMMAND_TIMEOUT_SECONDS = 120
COMMAND_TERMINATION_GRACE_SECONDS = 2


def capture_command(*args: str | Path, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command_args = [str(arg) for arg in args]
    command = shlex.join(command_args)
    process = subprocess.Popen(
        command_args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        env=env,
        start_new_session=True,
    )
    try:
        body, _ = process.communicate(timeout=COMMAND_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            body, _ = process.communicate(timeout=COMMAND_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            body, _ = process.communicate()
        details = f'\n{body}' if body else ''
        raise TimeoutError(f'{command} exceeded {COMMAND_TIMEOUT_SECONDS} seconds{details}') from None
    assert process.returncode is not None
    return subprocess.CompletedProcess(command_args, process.returncode, stdout=body)


def capture_frog(*args: str | Path, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return capture_command(FROGC, *args, env=env)


def check_frog_regressions() -> None:
    for artifact in CLI_BUILD_ARTIFACTS:
        artifact.unlink(missing_ok=True)
    try:
        result = capture_command(FROG_TESTS, FROGC, FROG_TEST_EXECUTABLE)
        assert result.returncode == 0, result.stdout
        assert result.stdout == '', result.stdout
    finally:
        for artifact in CLI_BUILD_ARTIFACTS:
            artifact.unlink(missing_ok=True)


def check_failed_build_policy() -> None:
    build_case = TMP_FS / 'build_policy'
    build_case.mkdir()
    build_main = build_case / 'main.frog'
    _ = shutil.copyfile(HOST_POLICY_FIXTURES / 'build_initial.frog', build_main)
    build_main_relative = build_main.relative_to(ROOT)

    initial_build = capture_frog('build', build_main_relative)
    assert initial_build.returncode == 0
    assert initial_build.stdout == ''

    built_c = build_main.with_suffix('.c')
    built_executable = build_main.with_suffix('.exe')
    original_c = built_c.read_bytes()
    original_executable = built_executable.read_bytes()

    _ = shutil.copyfile(HOST_POLICY_FIXTURES / 'build_updated.frog', build_main)
    fake_bin = build_case / 'fake-bin'
    fake_bin.mkdir()
    false_executable = shutil.which('false')
    assert false_executable is not None
    (fake_bin / 'gcc').symlink_to(false_executable)
    failing_environment = os.environ.copy()
    assert 'PATH' in failing_environment
    failing_environment['PATH'] = f'{fake_bin}{os.pathsep}{failing_environment["PATH"]}'

    failed_build = capture_frog('build', build_main_relative, env=failing_environment)
    assert failed_build.returncode != 0
    failed_c = built_c.read_bytes()
    assert failed_c != original_c
    assert built_executable.read_bytes() == original_executable

    updated_build = capture_frog('build', build_main_relative)
    assert updated_build.returncode == 0
    assert updated_build.stdout == ''
    assert built_c.read_bytes() == failed_c
    assert built_executable.read_bytes() != original_executable


def check_lexical_symlink_import() -> None:
    symlink_case = TMP_FS / 'symlink_root'
    _ = shutil.copytree(HOST_POLICY_FIXTURES / 'symlink_root', symlink_case)
    real_main = symlink_case / 'real' / 'main.frog'
    lexical_main = symlink_case / 'lexical' / 'main.frog'
    lexical_main.symlink_to(real_main)

    result = capture_frog('run', lexical_main.relative_to(ROOT))
    assert result.returncode == 0
    assert result.stdout == '42\n'


def check_slashless_compiler_policy() -> None:
    environment = os.environ.copy()
    assert 'PATH' in environment
    environment['PATH'] = f'{FROGC.parent}{os.pathsep}{environment["PATH"]}'

    help_result = capture_command('frogc', '--help', env=environment)
    assert help_result.returncode == 0
    assert help_result.stdout.startswith('Usage:\n')

    compile_result = capture_command('frogc', 'run', '-c', 'proc main -- do end', env=environment)
    assert compile_result.returncode == 1
    assert compile_result.stdout == 'frogc: standard library path is unavailable\n'


def main() -> None:
    arguments = sys.argv[1:]
    if arguments:
        if arguments == ['--frog-only']:
            check_frog_regressions()
            return
        raise SystemExit('usage: python -m test [--frog-only]')

    shutil.rmtree(TMP_FS, ignore_errors=True)
    TMP_FS.mkdir()
    try:
        check_failed_build_policy()
        check_lexical_symlink_import()
        check_slashless_compiler_policy()
    finally:
        shutil.rmtree(TMP_FS, ignore_errors=True)


if __name__ == '__main__':
    main()
