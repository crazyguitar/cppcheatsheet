#pragma once

extern int global_initialized;
extern int global_uninitialized;

int add(int a, int b);
static inline int square(int x) { return x * x; }
