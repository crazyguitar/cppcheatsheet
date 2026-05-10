// math.cppm - module interface unit
export module math;

export int add(int a, int b) { return a + b; }

export int multiply(int a, int b) { return a * b; }

// Not exported - internal to module
int helper() { return 42; }
