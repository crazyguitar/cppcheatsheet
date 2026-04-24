==========================
C/C++ Interview Cheatsheet
==========================

.. meta::
    :description lang=en: Curated C and C++ interview questions, each linking directly to the exact cheatsheet section that answers it. Covers RAII, smart pointers, move semantics, templates, concepts, the STL, modern C++ features, concurrency, and debugging tools.
    :keywords: C++ interview questions, C interview questions, RAII, smart pointers, unique_ptr, shared_ptr, move semantics, Rule of Five, templates, SFINAE, concepts, STL, vector, unordered_map, iterator invalidation, lambda, coroutine, modules, virtual function, undefined behavior, memory leak, Valgrind, AddressSanitizer, gdb, CMake

This page is a curated, question-indexed map into the rest of the cheatsheet.
Each entry below is a question you are likely to see in a C or C++ interview,
followed by a link that jumps directly to the section of the notes that
answers it. It is intentionally a navigation layer — the actual explanations,
code, and caveats live in the linked sections.

Use it two ways:

* **Drilling a topic:** pick a group (e.g. *Templates & Generics*) and walk
  every question in it.
* **Quick review before an interview:** read the questions, and for any you
  cannot confidently answer in one or two sentences, click through.

.. contents:: Topics
    :local:
    :depth: 1

Memory & Resource Management
============================

* What is RAII, and why is it the cornerstone of modern C++ resource
  management? :ref:`→ cpp/cpp_raii: RAII Wrapper <notes/cpp/cpp_raii:RAII Wrapper>`
* How does RAII interact with exception safety (basic / strong / nothrow
  guarantees)? :ref:`→ cpp/cpp_raii: Exception Safety Guarantees <notes/cpp/cpp_raii:Exception Safety Guarantees>`
* What happens when a constructor throws after partial resource acquisition?
  :ref:`→ cpp/cpp_raii: Constructor Failure and Exception Safety <notes/cpp/cpp_raii:Constructor Failure and Exception Safety>`
* Compare ``std::lock_guard``, ``std::unique_lock``, and ``std::scoped_lock``.
  :ref:`→ cpp/cpp_raii: Lock-Based RAII <notes/cpp/cpp_raii:Lock-Based RAII>`
* When does ``std::unique_ptr`` make sense vs. ``std::shared_ptr``?
  :ref:`→ cpp/cpp_smartpointers: std::unique_ptr: Exclusive Ownership <notes/cpp/cpp_smartpointers:std::unique_ptr\: Exclusive Ownership>`
* What problem does ``std::weak_ptr`` solve?
  :ref:`→ cpp/cpp_smartpointers: std::weak_ptr: Non-Owning Observer <notes/cpp/cpp_smartpointers:std::weak_ptr\: Non-Owning Observer>`
* Why prefer ``std::make_unique`` / ``std::make_shared`` over raw ``new``?
  :ref:`→ cpp/cpp_smartpointers: std::make_unique and std::make_shared <notes/cpp/cpp_smartpointers:std::make_unique and std::make_shared>`
* What are common smart-pointer pitfalls (e.g. cycles, double ownership)?
  :ref:`→ cpp/cpp_smartpointers: Common Pitfalls <notes/cpp/cpp_smartpointers:Common Pitfalls>`
* Why does memory alignment matter, and how is struct padding computed?
  :ref:`→ c/c_memory: Structure Alignment and Padding <notes/c/c_memory:Structure Alignment and Padding>`

Move Semantics & Value Categories
=================================

* What are the special member functions, and when are they implicitly defined
  or deleted? :ref:`→ cpp/cpp_move: Special Member Functions <notes/cpp/cpp_move:Special Member Functions>`
* Explain the Rule of Zero, Three, and Five.
  :ref:`→ cpp/cpp_move: Rule of Zero <notes/cpp/cpp_move:Rule of Zero>` ·
  :ref:`Three <notes/cpp/cpp_move:Rule of Three>` ·
  :ref:`Five <notes/cpp/cpp_move:Rule of Five>`
* What are lvalues, rvalues, xvalues, prvalues?
  :ref:`→ cpp/cpp_move: Value Categories <notes/cpp/cpp_move:Value Categories>`
* How does ``std::forward`` achieve perfect forwarding, and why is it needed?
  :ref:`→ cpp/cpp_move: Perfect Forwarding <notes/cpp/cpp_move:Perfect Forwarding>`
* What state is a moved-from object left in, and what can you safely do with
  it? :ref:`→ cpp/cpp_move: Moved-From State <notes/cpp/cpp_move:Moved-From State>`
* Why should move constructors typically be ``noexcept``?
  :ref:`→ cpp/cpp_move: Conditional noexcept <notes/cpp/cpp_move:Conditional noexcept>`
* ``emplace_back`` vs. ``push_back`` — when is there a real difference?
  :ref:`→ cpp/cpp_move: Emplace vs Insert <notes/cpp/cpp_move:Emplace vs Insert>`

Templates & Generics
====================

* How does template specialization differ from overloading?
  :ref:`→ cpp/cpp_template: Template Specialization <notes/cpp/cpp_template:Template Specialization>`
* What are variadic templates and fold expressions?
  :ref:`→ cpp/cpp_template: Variadic Templates and Parameter Packs <notes/cpp/cpp_template:Variadic Templates and Parameter Packs>` ·
  :ref:`Fold Expressions <notes/cpp/cpp_template:Fold Expressions (C++17)>`
* What is SFINAE, and how do C++20 concepts replace most of it?
  :ref:`→ cpp/cpp_template: SFINAE and Type Constraints <notes/cpp/cpp_template:SFINAE and Type Constraints>` ·
  :ref:`Defining Concepts <notes/cpp/cpp_concepts:Defining Concepts>`
* What is CRTP and when would you use it?
  :ref:`→ cpp/cpp_template: CRTP <notes/cpp/cpp_template:CRTP (Curiously Recurring Template Pattern)>`
* How does a ``requires`` clause constrain a template, and how does it compose
  with ``&&`` / ``||``?
  :ref:`→ cpp/cpp_requires: Basic Requires Clause <notes/cpp/cpp_requires:Basic Requires Clause>` ·
  :ref:`Conjunction <notes/cpp/cpp_requires:Conjunction (&&)>` ·
  :ref:`Disjunction <notes/cpp/cpp_requires:Disjunction (\|\|)>`
* ``requires`` vs. ``if constexpr`` — when to reach for which?
  :ref:`→ cpp/cpp_requires: Requires vs if constexpr <notes/cpp/cpp_requires:Requires vs if constexpr>`
* What are reference collapsing rules, and why do they matter for forwarding
  references? :ref:`→ cpp/cpp_basic: Reference Collapsing Rules <notes/cpp/cpp_basic:Reference Collapsing Rules>`
* How does template type deduction differ between by-value, by-ref, and
  forwarding parameters?
  :ref:`→ cpp/cpp_basic: Template Type Deduction <notes/cpp/cpp_basic:Template Type Deduction>`

Casting & Type Conversions
==========================

* Why prefer ``static_cast``, ``dynamic_cast``, ``const_cast``, and
  ``reinterpret_cast`` over C-style casts?
  :ref:`→ cpp/cpp_casting: C-Style Cast (Legacy Casting) <notes/cpp/cpp_casting:C-Style Cast (Legacy Casting)>`
* When is ``dynamic_cast`` safe, and when does it return null or throw?
  :ref:`→ cpp/cpp_casting: dynamic_cast <notes/cpp/cpp_casting:dynamic_cast\: Runtime Type-Safe Downcasting>`
* What does ``const_cast`` actually guarantee, and when is it undefined?
  :ref:`→ cpp/cpp_casting: const_cast <notes/cpp/cpp_casting:const_cast\: Removing or Adding const>`

Compile-Time Programming
========================

* ``const`` vs. ``constexpr`` vs. ``consteval`` vs. ``constinit`` — pick the
  right one.
  :ref:`→ cpp/cpp_constexpr: constexpr Functions <notes/cpp/cpp_constexpr:constexpr Functions>` ·
  :ref:`consteval <notes/cpp/cpp_constexpr:consteval\: Immediate Functions (C++20)>` ·
  :ref:`constinit <notes/cpp/cpp_constexpr:constinit\: Constant Initialization (C++20)>`
* How does ``if constexpr`` eliminate branches at compile time?
  :ref:`→ cpp/cpp_constexpr: constexpr if (C++17) <notes/cpp/cpp_constexpr:constexpr if (C++17)>`
* How has ``constexpr`` expanded across C++14, 17, 20, and 23?
  :ref:`→ cpp/cpp_constexpr: constexpr Evolution by Standard <notes/cpp/cpp_constexpr:constexpr Evolution by Standard>`

STL Containers & Iterators
==========================

* Compare ``std::vector``, ``std::deque``, and ``std::list`` — memory layout,
  iterator invalidation, complexity.
  :ref:`→ cpp/cpp_container: std::vector <notes/cpp/cpp_container:std\:\:vector>` ·
  :ref:`std::deque <notes/cpp/cpp_container:std\:\:deque>` ·
  :ref:`std::list <notes/cpp/cpp_container:std\:\:list>`
* ``std::map`` vs. ``std::unordered_map`` — trade-offs and when to use each.
  :ref:`→ cpp/cpp_container: Ordered vs Unordered Containers <notes/cpp/cpp_container:Ordered vs Unordered Containers>`
* What are the iterator invalidation rules for the common containers?
  :ref:`→ cpp/cpp_iterator: Iterator Invalidation <notes/cpp/cpp_iterator:Iterator Invalidation>`
* What are iterator categories, and why do algorithms care?
  :ref:`→ cpp/cpp_iterator: Iterator Categories <notes/cpp/cpp_iterator:Iterator Categories>`
* Walk through ``std::sort``, ``std::stable_sort``, ``std::partial_sort``, and
  ``std::nth_element`` — when would you pick each?
  :ref:`→ cpp/cpp_algorithm: Sorting <notes/cpp/cpp_algorithm:Sorting>`
* Difference between ``std::find`` and ``std::binary_search``?
  :ref:`→ cpp/cpp_algorithm: Searching <notes/cpp/cpp_algorithm:Searching>`
* What are C++20 ranges and views? Are views lazy?
  :ref:`→ cpp/cpp_iterator: C++20 Ranges Overview <notes/cpp/cpp_iterator:C++20 Ranges Overview>`

Strings
=======

* ``std::string`` vs. ``std::string_view`` — when is ``string_view`` dangerous?
  :ref:`→ cpp/cpp_string: std::string_view <notes/cpp/cpp_string:std\:\:string_view>`
* Why is naïve ``+=`` concatenation in a loop a performance trap?
  :ref:`→ cpp/cpp_string: String Concatenation Performance <notes/cpp/cpp_string:String Concatenation Performance>`

Modern C++ Features
===================

* What does each lambda capture mode mean: ``[]``, ``[=]``, ``[&]``,
  ``[this]``, init captures?
  :ref:`→ cpp/cpp_lambda: Lambda Syntax Overview <notes/cpp/cpp_lambda:Lambda Syntax Overview>` ·
  :ref:`Init Capture <notes/cpp/cpp_lambda:Init Capture (C++14)>`
* Why is a captureless lambda convertible to a function pointer?
  :ref:`→ cpp/cpp_lambda: Captureless Lambdas and Function Pointers <notes/cpp/cpp_lambda:Captureless Lambdas and Function Pointers>`
* What is a generic / template lambda?
  :ref:`→ cpp/cpp_lambda: Generic Lambdas (C++14) <notes/cpp/cpp_lambda:Generic Lambdas (C++14)>` ·
  :ref:`Template Lambdas <notes/cpp/cpp_lambda:Template Lambdas (C++20)>`
* What are C++20 coroutines, and what does ``co_await`` actually do?
  :ref:`→ cpp/cpp_coroutine: Coroutine Basics <notes/cpp/cpp_coroutine:Coroutine Basics>` ·
  :ref:`Promise Type <notes/cpp/cpp_coroutine:Promise Type>` ·
  :ref:`Awaiter <notes/cpp/cpp_coroutine:Awaiter>`
* Why would you write a generator with coroutines instead of a class?
  :ref:`→ cpp/cpp_coroutine: Generator <notes/cpp/cpp_coroutine:Generator>`
* What are C++20 modules, and what problem do they solve?
  :ref:`→ cpp/cpp_modules: Module Basics <notes/cpp/cpp_modules:Module Basics>`
* ``auto`` vs. ``decltype`` vs. ``decltype(auto)``?
  :ref:`→ cpp/cpp_basic: Auto Type Deduction <notes/cpp/cpp_basic:Auto Type Deduction>` ·
  :ref:`decltype(auto) <notes/cpp/cpp_basic:decltype(auto) Type Deduction>`
* What does uniform / brace initialization change about narrowing conversions?
  :ref:`→ cpp/cpp_basic: Uniform Initialization <notes/cpp/cpp_basic:Uniform Initialization (Brace Initialization)>`

C Language Essentials
=====================

* When does an array *not* decay to a pointer, and what are the signed/unsigned
  gotchas around pointer arithmetic?
  :ref:`→ cpp/cpp_basic: Pointer Arithmetic and Negative Indices <notes/cpp/cpp_basic:Pointer Arithmetic and Negative Indices>`
* What does ``restrict`` promise to the compiler?
  :ref:`→ c/c_basic: Restrict Qualifier <notes/c/c_basic:Restrict Qualifier>`
* Designated initializers and compound literals — how are they different?
  :ref:`→ c/c_basic: Designated Initializers <notes/c/c_basic:Designated Initializers>` ·
  :ref:`Compound Literals <notes/c/c_basic:Compound Literals>`
* What are flexible array members?
  :ref:`→ c/c_basic: Flexible Array Members <notes/c/c_basic:Flexible Array Members>`
* Why is ``_Generic`` useful, and how is it different from C++ overloading?
  :ref:`→ c/c_basic: _Generic Type Selection (C11) <notes/c/c_basic:\`\`_Generic\`\` Type Selection (C11)>`
* What new things did C23 bring (``nullptr``, ``constexpr``, ``typeof``,
  attributes, bit-precise ints)?
  :ref:`→ c/c_basic: typeof and auto (C23) <notes/c/c_basic:\`\`typeof\`\` and \`\`auto\`\` (C23)>` ·
  :ref:`nullptr <notes/c/c_basic:\`\`nullptr\`\` (C23)>` ·
  :ref:`constexpr <notes/c/c_basic:\`\`constexpr\`\` (C23)>`
* Why are macros dangerous, and what safer alternatives exist?
  :ref:`→ c/c_macro: Variadic Macros <notes/c/c_macro:Variadic Macros>`

Concurrency & OS
================

* How do you create threads in modern C++?
  :ref:`→ os/os_thread: Creating Threads <notes/os/os_thread:Creating Threads>`
* Mutex, condition variable, and read-write lock — when is each appropriate?
  :ref:`→ os/os_thread: Mutex Synchronization <notes/os/os_thread:Mutex Synchronization>` ·
  :ref:`Condition Variables <notes/os/os_thread:Condition Variables>` ·
  :ref:`Read-Write Locks <notes/os/os_thread:Read-Write Locks>`
* ``std::future`` / ``std::promise`` / ``std::async`` — what are the launch
  policies?
  :ref:`→ os/os_thread: C++ Futures and Async <notes/os/os_thread:C++ Futures and Async>` ·
  :ref:`Launch Policies <notes/os/os_thread:Launch Policies>`
* What is thread-local storage?
  :ref:`→ os/os_thread: Thread-Local Storage <notes/os/os_thread:Thread-Local Storage>`
* Process vs. thread — what is shared and what is not?
  :ref:`→ os/os_process: Fork and Wait <notes/os/os_process:Fork and Wait>` ·
  :ref:`Fork and Exec <notes/os/os_process:Fork and Exec>`

Debugging & Tools
=================

* How do you find memory leaks and invalid accesses with Valgrind?
  :ref:`→ debug/valgrind: Memcheck: Memory Errors <notes/debug/valgrind:Memcheck\: Memory Errors>` ·
  :ref:`Leak Types <notes/debug/valgrind:Leak Types>`
* What do AddressSanitizer, ThreadSanitizer, and UBSan each catch?
  :ref:`→ debug/sanitizers: AddressSanitizer (ASan) <notes/debug/sanitizers:AddressSanitizer (ASan)>` ·
  :ref:`ThreadSanitizer <notes/debug/sanitizers:ThreadSanitizer (TSan)>` ·
  :ref:`UndefinedBehaviorSanitizer <notes/debug/sanitizers:UndefinedBehaviorSanitizer (UBSan)>`
* How do you inspect a core dump with gdb after a segfault?
  :ref:`→ debug/gdb: Debugging Crashes <notes/debug/gdb:Debugging Crashes>`
* What are breakpoints vs. watchpoints in gdb?
  :ref:`→ debug/gdb: Breakpoints <notes/debug/gdb:Breakpoints>` ·
  :ref:`Watchpoints <notes/debug/gdb:Watchpoints>`
* Minimal modern CMake project — what are the must-have commands?
  :ref:`→ cpp/cpp_cmake: Minimal Project <notes/cpp/cpp_cmake:Minimal Project>` ·
  :ref:`Modern C++ Standard Setting <notes/cpp/cpp_cmake:Modern C++ Standard Setting>`

See Also
========

If a question above is not covered, the top-level indices are the best next
stop:

* :doc:`../c/index` — C language reference
* :doc:`../cpp/index` — Modern C++ reference
* :doc:`../os/index` — OS and systems programming
* :doc:`../debug/index` — Debugging and profiling tools
