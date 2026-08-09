#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int frog_compiler_main(void);

const char *const gcc_flags[] = {
    "-std=c11", "-pedantic", "-Wall", "-Wextra", "-Wconversion", "-Werror", "-O2", "-x", "c", NULL,
};

void error(const char *message) { fprintf(stderr, "frogc: %s\n", message); }

void error_path(const char *message, const char *path) { fprintf(stderr, "frogc: %s: %s\n", message, path); }

int usage_error(const char *message) {
    error(message);
    fputs("Try `frogc --help`.\n", stderr);
    return 2;
}

void usage(void) {
    fputs("Usage:\n"
          "  frogc < source.frog > source.c\n"
          "  frogc <command> [options]\n\n"
          "Commands:\n"
          "  run [-c CODE | FILE]       compile and run Frog source\n"
          "  build [-o FILE] [-r] FILE  compile Frog source to a binary\n",
          stdout);
}

char *duplicate_range(const char *text, size_t length) {
    char *copy = malloc(length + 1U);
    if (copy == NULL) return NULL;
    memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

char *parent_path(const char *path) {
    const char *slash = strrchr(path, '/');
    if (slash == NULL) return duplicate_range(".", 1U);
    if (slash == path) return duplicate_range("/", 1U);
    return duplicate_range(path, (size_t)(slash - path));
}

char *replace_suffix(const char *path, const char *suffix) {
    const char *slash = strrchr(path, '/');
    const char *dot = strrchr(path, '.');
    const char *base = slash == NULL ? path : slash + 1;
    size_t prefix = (dot != NULL && dot > base) ? (size_t)(dot - path) : strlen(path);
    size_t suffix_length = strlen(suffix);
    char *result = malloc(prefix + suffix_length + 1U);
    if (result == NULL) return NULL;
    memcpy(result, path, prefix);
    memcpy(result + prefix, suffix, suffix_length + 1U);
    return result;
}

int wait_status(pid_t child) {
    int status;
    for (;;) {
        if (waitpid(child, &status, 0) >= 0) break;
        if (errno != EINTR) { error("unable to wait for child"); return 1; }
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
}

void reset_child_signals(void) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGINT, &action, NULL);
    (void)sigaction(SIGTERM, &action, NULL);
    (void)sigaction(SIGPIPE, &action, NULL);
    (void)sigaction(SIGHUP, &action, NULL);
}

int write_all(int fd, const unsigned char *bytes, size_t length) {
    while (length != 0U) {
        ssize_t written = write(fd, bytes, length);
        if (written < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        bytes += (size_t)written;
        length -= (size_t)written;
    }
    return 0;
}

int read_source(const char *path, unsigned char **bytes, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) { error_path("source file not found", path); return 1; }
    if (fseek(file, 0, SEEK_END) != 0) goto failed;
    long end = ftell(file);
    if (end < 0) goto failed;
    if (fseek(file, 0, SEEK_SET) != 0) goto failed;
    *length = (size_t)end;
    *bytes = malloc(*length == 0U ? 1U : *length);
    if (*bytes == NULL) goto failed;
    if (fread(*bytes, 1U, *length, file) != *length) { free(*bytes); *bytes = NULL; goto failed; }
    if (fclose(file) != 0) { free(*bytes); *bytes = NULL; error_path("unable to read", path); return 1; }
    return 0;
failed:
    error_path("unable to read", path);
    (void)fclose(file);
    return 1;
}

int compile_frog(const unsigned char *source, size_t length, const char *directory, const char *output) {
    int input[2] = {-1, -1};
    int output_fd = open(output, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int directory_fd = open(directory, O_RDONLY | O_DIRECTORY);
    if (pipe(input) != 0 || output_fd < 0 || directory_fd < 0) {
        error("unable to prepare compiler input or output");
        if (input[0] >= 0) (void)close(input[0]);
        if (input[1] >= 0) (void)close(input[1]);
        if (output_fd >= 0) (void)close(output_fd);
        if (directory_fd >= 0) (void)close(directory_fd);
        return 1;
    }
    pid_t child = fork();
    if (child < 0) { error("unable to fork compiler"); (void)close(input[0]); (void)close(input[1]); (void)close(output_fd); (void)close(directory_fd); return 1; }
    if (child == 0) {
        reset_child_signals();
        if (dup2(input[0], STDIN_FILENO) < 0 || dup2(output_fd, STDOUT_FILENO) < 0 || fchdir(directory_fd) < 0) {
            int saved_errno = errno;
            dprintf(STDERR_FILENO, "frogc: unable to prepare compiler child: %s\n", strerror(saved_errno));
            _exit(1);
        }
        (void)close(input[0]); (void)close(input[1]); (void)close(output_fd); (void)close(directory_fd);
        int result = frog_compiler_main();
        if (fflush(stdout) != 0) result = 1;
        _exit(result);
    }
    (void)close(input[0]); (void)close(output_fd); (void)close(directory_fd);
    struct sigaction ignored, previous;
    memset(&ignored, 0, sizeof(ignored)); ignored.sa_handler = SIG_IGN; sigemptyset(&ignored.sa_mask);
    if (sigaction(SIGPIPE, &ignored, &previous) != 0) { error("unable to ignore SIGPIPE while sending compiler input"); (void)close(input[1]); (void)wait_status(child); return 1; }
    int write_result = write_all(input[1], source, length);
    int close_result = close(input[1]);
    (void)sigaction(SIGPIPE, &previous, NULL);
    int child_result = wait_status(child);
    if (write_result != 0 || close_result != 0) { error("unable to send source to compiler"); return 1; }
    return child_result;
}

int compile_c(const char *source, const char *executable) {
    char *args[14];
    size_t index = 0U;
    args[index++] = "gcc";
    for (size_t flag = 0U; gcc_flags[flag] != NULL; ++flag) args[index++] = (char *)gcc_flags[flag];
    args[index++] = (char *)source;
    args[index++] = "-o";
    args[index++] = (char *)executable;
    args[index] = NULL;
    pid_t child = fork();
    if (child < 0) { error("unable to fork gcc"); return 1; }
    if (child == 0) { reset_child_signals(); execvp("gcc", args); dprintf(STDERR_FILENO, "frogc: unable to run gcc: %s\n", strerror(errno)); _exit(127); }
    return wait_status(child);
}

int run_program(const char *executable) {
    pid_t child = fork();
    if (child < 0) { error("unable to fork executable"); return 1; }
    if (child == 0) { char *const args[] = {(char *)executable, NULL}; reset_child_signals(); execv(executable, args); dprintf(STDERR_FILENO, "frogc: unable to run %s: %s\n", executable, strerror(errno)); _exit(127); }
    return wait_status(child);
}

char *path_identity(const char *path) {
    char *parent = parent_path(path);
    const char *name = strrchr(path, '/');
    name = name == NULL ? path : name + 1;
    if (parent == NULL) return NULL;
    char resolved[PATH_MAX];
    if (realpath(parent, resolved) == NULL) { free(parent); return NULL; }
    size_t length = strlen(resolved) + strlen(name) + 2U;
    char *identity = malloc(length);
    if (identity != NULL) (void)snprintf(identity, length, "%s/%s", resolved, name);
    free(parent);
    return identity;
}

int paths_alias(const char *first, const char *second) {
    struct stat a, b;
    if (stat(first, &a) == 0 && stat(second, &b) == 0 && a.st_dev == b.st_dev && a.st_ino == b.st_ino) return 1;
    char *a_identity = path_identity(first);
    char *b_identity = path_identity(second);
    int aliases = a_identity != NULL && b_identity != NULL && strcmp(a_identity, b_identity) == 0;
    free(a_identity); free(b_identity);
    return aliases;
}

int ensure_build_directory(void) {
    if (mkdir("build", 0777) != 0 && errno != EEXIST) { error("unable to create build directory"); return 1; }
    struct stat info;
    if (stat("build", &info) != 0 || !S_ISDIR(info.st_mode)) { error("build path is not a directory"); return 1; }
    return 0;
}

int run_source(const unsigned char *source, size_t length, const char *directory) {
    if (ensure_build_directory() != 0) return 1;
    if (compile_frog(source, length, directory, "build/frog-run.c") != 0) return 1;
    if (compile_c("build/frog-run.c", "build/frog-run.exe") != 0) return 1;
    return run_program("build/frog-run.exe");
}

int run_file(const char *path) {
    unsigned char *source = NULL; size_t length = 0U; char *directory = parent_path(path);
    if (directory == NULL || read_source(path, &source, &length) != 0) { free(directory); return 1; }
    int result = run_source(source, length, directory); free(source); free(directory); return result;
}

int build_file(const char *path, const char *output, bool should_run) {
    unsigned char *source = NULL; size_t length = 0U; char *c_file = replace_suffix(path, ".c"); char *executable = output == NULL ? replace_suffix(path, ".exe") : duplicate_range(output, strlen(output)); char *directory = parent_path(path);
    int failure_status = 1;
    if (c_file == NULL || executable == NULL || directory == NULL || read_source(path, &source, &length) != 0) goto failed;
    if (paths_alias(path, c_file)) {
        failure_status = usage_error("generated C path aliases the source file");
        goto failed;
    }
    if (paths_alias(path, executable)) {
        failure_status = usage_error("executable path aliases the source file");
        goto failed;
    }
    if (paths_alias(c_file, executable)) {
        failure_status = usage_error("executable path aliases the generated C file");
        goto failed;
    }
    if (paths_alias(c_file, "/proc/self/exe") || paths_alias(executable, "/proc/self/exe")) {
        failure_status = usage_error("build output aliases the running compiler");
        goto failed;
    }
    int result = compile_frog(source, length, directory, c_file);
    if (result == 0) result = compile_c(c_file, executable);
    if (result == 0 && should_run) result = run_program(executable);
    free(source); free(c_file); free(executable); free(directory); return result;
failed:
    free(source); free(c_file); free(executable); free(directory); return failure_status;
}

int run_command(int argc, char **argv) {
    if (argc == 1 && (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0)) { fputs("Usage: frogc run [-c CODE | FILE]\n", stdout); return 0; }
    if (argc == 0) return usage_error("run requires a source file or -c CODE");
    if (strcmp(argv[0], "-c") == 0) { if (argc != 2) return usage_error("run -c requires exactly one CODE argument"); char cwd[PATH_MAX]; if (getcwd(cwd, sizeof(cwd)) == NULL) { error("unable to determine current directory"); return 1; } return run_source((const unsigned char *)argv[1], strlen(argv[1]), cwd); }
    if (argv[0][0] == '-') { char message[256]; (void)snprintf(message, sizeof(message), "unknown run option: %s", argv[0]); return usage_error(message); }
    if (argc != 1) return usage_error("run accepts exactly one source file");
    return run_file(argv[0]);
}

int build_command(int argc, char **argv) {
    const char *output = NULL; bool should_run = false; int index = 0;
    while (index < argc && argv[index][0] == '-') { const char *option = argv[index++]; if (strcmp(option, "-h") == 0 || strcmp(option, "--help") == 0) { fputs("Usage: frogc build [-o FILE] [-r] FILE\n", stdout); return 0; } if (strcmp(option, "-r") == 0) { should_run = true; continue; } if (strcmp(option, "-o") == 0) { if (index == argc) return usage_error("build -o requires an output file"); output = argv[index++]; continue; } char message[256]; (void)snprintf(message, sizeof(message), "unknown build option: %s", option); return usage_error(message); }
    if (argc - index != 1) return usage_error("build requires exactly one source file");
    return build_file(argv[index], output, should_run);
}

#ifndef FROG_CLI_NO_MAIN
int main(int argc, char **argv) {
    if (argc == 1) return frog_compiler_main();
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) { usage(); return 0; }
    if (strcmp(argv[1], "run") == 0) return run_command(argc - 2, argv + 2);
    if (strcmp(argv[1], "build") == 0) return build_command(argc - 2, argv + 2);
    char message[256]; (void)snprintf(message, sizeof(message), "unknown command: %s", argv[1]); return usage_error(message);
}
#endif
