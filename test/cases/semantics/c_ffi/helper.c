int ffi_test_answer(void) {
    return 42;
}

int p0(void) {
    return 9;
}

int ffi_test_mix(int number, int truth, void *pointer) {
    return number * 100 + (truth != 0 ? 10 : 0) + (pointer != (void *)0 ? 1 : 0);
}

int ffi_test_truth(int value) {
    return value * 4;
}

void *ffi_test_identity(void *pointer) {
    return pointer;
}
