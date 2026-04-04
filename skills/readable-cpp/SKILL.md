---
name: readable-cpp
description: Readable C/C++/Rust code rules inspired by The Art of Readable Code. Use when writing, reviewing, or refactoring C, C++, or Rust code. Enforces short functions, flat control flow, clear naming, readable structure, and idiomatic patterns.
---

# Readable C/C++/Rust Rules (/readable-cpp)

Apply these rules when writing, reviewing, or refactoring C, C++, or Rust code. Inspired by *The Art of Readable Code* by Dustin Boswell and Trevor Foucher.

**Core principle: Code should be easy to understand.** The time it takes someone else (or future you) to understand the code is the ultimate metric.

## 1. Keep Functions Short and Focused

- A function should do **one thing**. If you can describe what it does with "and", split it.
- Aim for functions that fit on one screen (~15-25 lines). If it's longer, extract sub-tasks.
- Each function should operate at a **single level of abstraction** — don't mix high-level logic with low-level details in the same function.

## 2. Flatten Control Flow — No Deep Nesting

- **Never nest more than 2 levels deep.** If you have a loop inside a loop, or an `if` inside a loop inside an `if`, extract the inner block into a helper function with a descriptive name.
- Use **early returns / guard clauses** to handle edge cases at the top, keeping the main logic flat.
- Prefer `continue` or `break` to skip iterations rather than wrapping the body in a conditional.
- Replace complex conditionals with well-named helper functions or variables that explain the intent.

```cpp
// Bad: nested and hard to follow
for (auto& user : users) {
    if (user.is_active()) {
        for (auto& order : user.orders()) {
            if (order.is_pending()) {
                process(order);
            }
        }
    }
}

// Good: flat, each function name explains what it does
auto active_users = get_active_users(users);
for (auto& user : active_users) {
    process_pending_orders(user.orders());
}
```

## 3. Name Things Clearly

- **Pack information into names.** Use specific, concrete words — `fetch_page` not `get`, `num_retries` not `n`.
- **Avoid generic names** like `tmp`, `data`, `result`, `val`, `info`, `handle` — unless the scope is tiny (2-3 lines).
- **Use names that can't be misconstrued.** If a range is inclusive, say `max_items` not `limit`. If a boolean, use `is_`, `has_`, `should_`, `can_` prefixes.
- **Match the name length to the scope.** Short names for small scopes, descriptive names for wide scopes.
- **Don't use abbreviations** unless they're universally understood (`num`, `max`, `min`, `err` are fine; `svc_mgr_cfg` is not).

## 4. Make Control Flow Easy to Follow

- Put the **changing/interesting value on the left** side of comparisons: `if (length > 10)` not `if (10 < length)`.
- Order `if/else` blocks: **positive case first**, simpler case first, or the more interesting case first.
- Minimize the number of variables the reader has to track. Reduce the **mental footprint** of each block.
- Avoid deeply nested ternary operators — if it's not immediately obvious, use an `if/else`.

## 5. Break Down Giant Expressions

- Use **explaining variables** to break complex expressions into named pieces.
- Use **summary variables** to capture a long expression that's used more than once.
- Apply **De Morgan's laws** to simplify negated boolean expressions.

```cpp
// Bad
if (!(age >= 18 && has_id && !is_banned)) {
    deny();
}

// Good
bool is_eligible = age >= 18 && has_id && !is_banned;
if (!is_eligible) {
    deny();
}
```

## 6. Extract Unrelated Subproblems

- If a block of code is solving a **subproblem unrelated to the main goal** of the function, extract it.
- The helper function should be **pure and self-contained** — it shouldn't need to know about the calling context.
- This is the single most effective way to improve readability: separate *what* you're doing from *how*.

## 7. One Task at a Time

- Each section of code should do **one task**. If a function is doing parsing AND validation AND transformation, split them into separate steps.
- List the tasks a function does. If there's more than one, reorganize so each task is in its own block or function.

## 8. Reduce Variable Scope

- **Declare variables close to where they're used.** Don't declare at the top of a function if it's only used 30 lines later.
- **Minimize the "live time" of a variable** — the fewer lines between its assignment and last use, the easier it is to follow.
- **Prefer write-once variables.** Variables that are assigned once and never modified are easier to reason about.
- **Eliminate unnecessary variables.** If a variable is used only once and doesn't clarify anything, inline it.

## 9. No Magic Numbers or Strings

- Replace **magic numbers and strings** with named constants: `if (retries > MAX_RETRIES)` not `if (retries > 3)`.
- If a value has meaning, give it a name. The name documents the intent.
- Group related constants together.

## 10. Fewer Function Arguments

- Aim for **3 or fewer arguments** per function. More than that is a smell.
- Group related arguments into a **struct, class, or tuple**.
- If a function needs many config-like options, pass a single config/options object.
- Boolean flag arguments are a sign the function does two things — split it instead.

## 11. Consistency

- If the codebase does something one way, **do it the same way**. Don't mix styles.
- Consistent naming patterns, consistent structure, consistent error handling.
- When joining an existing codebase, **match the existing conventions** even if you'd prefer a different style.
- Surprise is the enemy of readability — predictable code is readable code.

## 12. Write Less Code

- The best code is **no code at all**. Question whether a feature is truly needed before implementing.
- **Don't over-engineer.** Solve the problem at hand, not hypothetical future problems.
- Remove dead code. Commented-out code is dead code.
- Use standard libraries before writing custom solutions.

## 13. Comments: Explain Why, Not What

- Don't comment **what** the code does — the code already says that. Comment **why** it does it.
- Comment **flaws and workarounds**: `// TODO:`, `// HACK:`, `// XXX:` with explanation.
- Comment **surprising behavior** or non-obvious decisions — things where a reader would ask "why?".
- **Don't comment bad code — rewrite it.** If you need a comment to explain what a block does, extract it into a well-named function instead.

---

# C-Specific Rules

## 14. RAII-Like Patterns with goto Cleanup

- In C, use the **goto cleanup pattern** for resource management — allocate at the top, clean up at a single labeled block at the bottom.
- Never scatter `free()` calls across multiple return paths. A single cleanup section is easier to audit.
- Use `__attribute__((cleanup))` (GCC/Clang) when available for automatic cleanup.

```c
// Good: single cleanup path
int process_file(const char *path) {
    int ret = -1;
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    char *buf = malloc(BUF_SIZE);
    if (!buf) goto cleanup_file;

    // ... do work ...
    ret = 0;

cleanup_buf:
    free(buf);
cleanup_file:
    fclose(fp);
    return ret;
}
```

## 15. Use `const` Liberally

- Mark pointers `const` when the function doesn't modify the pointed-to data: `const char *msg`.
- Mark local variables `const` when they don't change after initialization.
- This documents intent and helps the compiler catch mistakes.

## 16. Prefer Sized Types for Data Structures

- Use `<stdint.h>` types (`uint32_t`, `int64_t`) for data that crosses boundaries (files, network, hardware).
- Use `size_t` for sizes and counts, `ptrdiff_t` for pointer differences.
- Use `int` and `unsigned` for simple loop counters and local arithmetic.

## 17. Defensive Macro Hygiene

- Wrap macro bodies in `do { ... } while(0)` for statement-like macros.
- Parenthesize all macro parameters: `#define SQUARE(x) ((x) * (x))`.
- Prefer `static inline` functions over macros when possible (type safety, debuggability).
- Use `_Generic` (C11) for type-safe "overloading" instead of macro tricks.

---

# C++-Specific Rules

## 18. Use RAII for All Resources

- Every resource (memory, file handles, locks, sockets) should be owned by an RAII object.
- Use `std::unique_ptr` for exclusive ownership, `std::shared_ptr` only when shared ownership is genuinely needed.
- Write custom RAII wrappers for non-standard resources (e.g., C library handles).
- Never use raw `new`/`delete` in application code — let smart pointers and containers handle it.

## 19. Prefer Value Semantics and Move

- Pass small objects by value, large objects by `const&`.
- Return objects by value — rely on RVO/NRVO and move semantics.
- Implement move constructors/assignment for types that own resources.
- Use `std::move` only when you truly want to transfer ownership — don't `std::move` from things you'll use again.

## 20. Use Modern C++ Over C Idioms

- Use `std::array` over C arrays, `std::string` over `char*`, `std::vector` over `malloc`/`realloc`.
- Use `std::optional` over sentinel values, `std::variant` over type-unsafe unions.
- Use range-based `for` loops: `for (const auto& item : container)`.
- Use structured bindings (C++17): `auto [key, value] = *map.begin();`.
- Use `std::format` (C++20) or `fmt::format` over `sprintf` / string concatenation.

## 21. Templates: Keep It Simple

- Use concepts (C++20) to constrain templates — errors become readable.
- Prefer `if constexpr` over SFINAE when possible.
- Don't write template metaprogramming unless the benefit is clear and the team can maintain it.
- A non-template solution that's slightly less generic is often better than a template solution nobody understands.

## 22. Use `constexpr` and `const` Aggressively

- Mark functions `constexpr` when they can be evaluated at compile time.
- Use `constexpr` variables instead of `#define` for constants.
- Use `const` on member functions that don't modify state.
- `consteval` (C++20) for functions that *must* be compile-time evaluated.

## 23. Error Handling: Pick One Pattern

- Use exceptions for truly exceptional conditions, `std::expected` (C++23) or `std::optional` for expected failures.
- Don't mix error codes and exceptions in the same layer.
- If using exceptions, make them specific — derive from `std::runtime_error`, not `std::exception`.
- Use `noexcept` on functions that cannot throw (destructors, move operations).

---

# Rust-Specific Rules

## 24. Embrace the Ownership Model

- Don't fight the borrow checker — redesign your data flow instead.
- Prefer passing references (`&T`, `&mut T`) over cloning. Clone only when ownership transfer is genuinely needed.
- Use lifetimes explicitly only when the compiler can't infer them — don't annotate unnecessarily.
- Prefer `&str` over `String` in function parameters when you don't need ownership.

## 25. Use Iterators and Combinators

- Prefer iterator chains (`.iter().filter().map().collect()`) over manual loops with indices.
- Use `for item in &collection` instead of `for i in 0..collection.len()`.
- Use `enumerate()`, `zip()`, `chain()`, `chunks()` — the iterator API is rich.
- Avoid `.unwrap()` in production code — use `?`, `unwrap_or`, `unwrap_or_else`, or pattern matching.

```rust
// Bad: manual indexing
let mut names = Vec::new();
for i in 0..users.len() {
    if users[i].is_active {
        names.push(users[i].name.clone());
    }
}

// Good: idiomatic iterator chain
let names: Vec<_> = users.iter()
    .filter(|u| u.is_active)
    .map(|u| u.name.clone())
    .collect();
```

## 26. Use Enums and Pattern Matching

- Use `enum` with data variants instead of class hierarchies or tagged unions.
- Use `match` exhaustively — the compiler ensures you handle all cases.
- Use `if let` / `while let` for single-variant matching instead of full `match`.
- Prefer `Result<T, E>` over panicking — make errors part of the type signature.

## 27. Leverage the Type System

- Use **newtype wrappers** (`struct UserId(u64)`) to prevent mixing up same-typed values.
- Use `Option<T>` instead of sentinel values or null pointers.
- Use `#[must_use]` on functions whose return values shouldn't be ignored.
- Prefer `From`/`Into` traits for type conversions over manual conversion functions.

## 28. Module Organization

- Keep `pub` surfaces small — expose only what's needed.
- Use `pub(crate)` for crate-internal visibility instead of full `pub`.
- Group related types and functions in modules — one concept per module.
- Re-export key types at the crate root for ergonomic imports.
