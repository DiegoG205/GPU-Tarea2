#include "kernel.h"

void vec_sum(int* a, int* b, int* c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}
