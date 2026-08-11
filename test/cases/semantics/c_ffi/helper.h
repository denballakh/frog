#ifndef FROG_TEST_C_FFI_HELPER_H
#define FROG_TEST_C_FFI_HELPER_H

int ffi_test_answer(void);
int p0(void);
int ffi_test_mix(int number, int truth, void *pointer);
int ffi_test_truth(int value);
void *ffi_test_identity(void *pointer);

#endif
