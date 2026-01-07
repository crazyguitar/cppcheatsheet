======================
Bash Basic Cheatsheet
======================

.. meta::
   :description: Comprehensive Bash scripting reference covering variables, arrays, string manipulation, conditionals, loops, functions, regex, text processing, and error handling for Linux/Unix shell programming.
   :keywords: Bash scripting, shell programming, Linux commands, Bash variables, Bash arrays, string manipulation, regex, awk, sed, grep, error handling, argument parsing

.. contents:: Table of Contents
    :backlinks: none

This cheatsheet provides a comprehensive reference for Bash shell scripting, covering
essential concepts from basic variable expansion to advanced text processing and error
handling. Each section includes practical examples that demonstrate real-world usage
patterns commonly encountered in automation scripts and system administration tasks.

An interactive demo script is available at `src/bash/bash.sh <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/bash/bash.sh>`_
to help you experiment with the concepts covered in this cheatsheet. The script provides
colorized output showing both the commands and their results, making it easier to
understand how each feature works.

.. code-block:: bash

    ./src/bash/bash.sh           # Run all demos
    ./src/bash/bash.sh strings   # Run specific section (e.g., strings, arrays, regex)
    ./src/bash/bash.sh --help    # Show all available sections

Special Parameters
------------------

Bash provides special parameters that give access to script arguments, process information,
and command exit status. These read-only variables are automatically set by the shell and
are essential for writing flexible, reusable scripts that can adapt to different inputs
and execution contexts.

.. code-block:: bash

    $*    # all positional params as "$1$2..."
    $@    # all positional params as "$1" "$2" ...
    $#    # number of positional params
    $$    # current process PID
    $?    # exit status of last command
    $0    # script name
    $!    # PID of last background command

Example:

.. code-block:: bash

    foo() {
      echo "All args: $@"
      echo "Arg count: $#"
      echo "Script: $0"
    }
    foo "a" "b" "c"

Set Positional Parameters
-------------------------

The ``set`` builtin can reassign positional parameters, which is useful for parsing
command output or restructuring arguments within a script. This technique allows you
to split strings into separate arguments or reset the argument list entirely for
subsequent processing.

.. code-block:: bash

    set -- a b c
    echo "$1" "$2" "$3"  # a b c

Brace Expansion
---------------

Brace expansion generates arbitrary strings before any other expansion occurs, making it
a powerful tool for creating multiple files, directories, or generating sequences. Unlike
globs, brace expansion does not depend on existing files and always produces output
regardless of filesystem state.

.. code-block:: bash

    echo foo.{pdf,txt,png}     # foo.pdf foo.txt foo.png
    echo {1..5}                # 1 2 3 4 5
    echo {a..z}                # a b c ... z
    echo {01..10}              # 01 02 03 ... 10
    mkdir -p project/{src,bin,doc}

Globs (Pattern Matching)
------------------------

Glob patterns enable filename expansion using wildcards. Unlike regular expressions,
globs are expanded by the shell before command execution and match against actual
filenames in the filesystem. Understanding the difference between ``*`` (any string),
``?`` (single character), and ``[]`` (character class) is fundamental to shell scripting.

.. code-block:: bash

    ls *.txt           # match any string
    ls file?.txt       # match single character
    ls file[abc].txt   # match a, b, or c
    ls file[a-z].txt   # match range a-z
    ls file[!0-9].txt  # match non-digit

POSIX character classes:

.. code-block:: bash

    [[:alnum:]]   # alphanumeric
    [[:alpha:]]   # alphabetic
    [[:digit:]]   # digits
    [[:space:]]   # whitespace
    [[:upper:]]   # uppercase
    [[:lower:]]   # lowercase

Variable Expansion
------------------

Parameter expansion provides powerful mechanisms for default values, substring extraction,
and variable introspection without spawning external processes. These expansions are
evaluated by the shell itself, making them significantly faster than calling utilities
like ``cut`` or ``sed`` for simple string operations.

.. code-block:: bash

    ${var:-default}    # use default if var unset/empty
    ${var:=default}    # assign default if var unset/empty
    ${var:+alt}        # use alt if var is set
    ${var:?error}      # exit with error if var unset/empty

    ${#var}            # string length
    ${var:offset}      # substring from offset
    ${var:offset:len}  # substring with length

    ${!prefix*}        # names matching prefix
    ${!prefix@}        # names matching prefix (quoted)

String Manipulation
-------------------

Bash provides built-in string operations for length calculation, substring extraction,
and case conversion. These native operations eliminate the need for external tools like
``cut``, ``tr``, or ``awk`` in many common scenarios, resulting in faster execution and
simpler scripts.

.. code-block:: bash

    foo="hello world"

    ${#foo}            # 11 (length)
    ${foo:6}           # world (from offset 6)
    ${foo:0:5}         # hello (first 5 chars)
    ${foo: -5}         # world (last 5 chars, note space)

    ${foo^^}           # HELLO WORLD (uppercase)
    ${foo,,}           # hello world (lowercase)
    ${foo^}            # Hello world (capitalize first)

Pattern Removal
---------------

Pattern removal operators strip matching prefixes or suffixes from variable values. The
``#`` operator removes from the front while ``%`` removes from the back. Single characters
perform shortest match; doubled characters perform longest (greedy) match. These operators
are particularly useful for path manipulation and file extension handling.

.. code-block:: bash

    path="/home/user/file.tar.gz"

    ${path#*/}         # home/user/file.tar.gz (remove shortest from front)
    ${path##*/}        # file.tar.gz (remove longest from front)
    ${path%.*}         # /home/user/file.tar (remove shortest from back)
    ${path%%.*}        # /home/user/file (remove longest from back)

Pattern Substitution
--------------------

Pattern substitution replaces matched patterns within variable values, providing
functionality similar to ``sed`` but without spawning a subprocess. Single slash replaces
the first occurrence; double slash replaces all occurrences. Anchors ``#`` and ``%``
restrict matching to the beginning or end of the string.

.. code-block:: bash

    str="foo bar foo"

    ${str/foo/baz}     # baz bar foo (replace first)
    ${str//foo/baz}    # baz bar baz (replace all)
    ${str/#foo/baz}    # baz bar foo (replace if starts with)
    ${str/%foo/baz}    # foo bar baz (replace if ends with)

Arrays
------

Indexed arrays store ordered lists of values accessible by numeric index. Arrays in Bash
are sparse, meaning indices need not be contiguous. They support efficient iteration,
element counting, and dynamic growth through append operations.

.. code-block:: bash

    arr=(a b c d)          # declare array
    arr+=(e f)             # append elements

    echo ${arr[0]}         # first element
    echo ${arr[@]}         # all elements
    echo ${#arr[@]}        # array length
    echo ${!arr[@]}        # all indices

    unset arr[1]           # remove element

Associative Arrays (Dictionary)
-------------------------------

Associative arrays, available in Bash 4.0 and later, provide key-value storage enabling
dictionary-like data structures. They must be explicitly declared with ``declare -A``
before use. Common applications include configuration management, caching computed values,
and building lookup tables for efficient data retrieval.

.. code-block:: bash

    declare -A dict
    dict=(["foo"]="FOO" ["bar"]="BAR")
    dict["baz"]="BAZ"

    echo ${dict[foo]}      # FOO
    echo ${!dict[@]}       # all keys
    echo ${dict[@]}        # all values

    # check if key exists
    [[ -v dict[foo] ]] && echo "exists"

    unset dict[foo]        # remove key

Conditionals
------------

Conditional expressions test file attributes, compare strings, and evaluate numeric
relationships. The ``[[ ]]`` construct is preferred over ``[ ]`` for its enhanced
features including pattern matching, regex support, and safer handling of empty variables.
Always quote variables inside conditions to prevent word splitting issues.

File tests:

.. code-block:: bash

    [[ -e file ]]    # exists
    [[ -f file ]]    # regular file
    [[ -d dir ]]     # directory
    [[ -r file ]]    # readable
    [[ -w file ]]    # writable
    [[ -x file ]]    # executable
    [[ -s file ]]    # size > 0
    [[ -L file ]]    # symbolic link

String tests:

.. code-block:: bash

    [[ -z "$str" ]]        # empty string
    [[ -n "$str" ]]        # non-empty string
    [[ "$a" == "$b" ]]     # equal
    [[ "$a" != "$b" ]]     # not equal
    [[ "$a" < "$b" ]]      # less than (lexicographic)
    [[ "$a" =~ regex ]]    # regex match

Numeric tests:

.. code-block:: bash

    [[ $a -eq $b ]]   # equal
    [[ $a -ne $b ]]   # not equal
    [[ $a -lt $b ]]   # less than
    [[ $a -le $b ]]   # less or equal
    [[ $a -gt $b ]]   # greater than
    [[ $a -ge $b ]]   # greater or equal

Arithmetic
----------

Bash supports integer arithmetic through ``(( ))`` for evaluation and ``$(( ))`` for
expansion. These constructs allow C-style syntax including increment/decrement operators,
compound assignment, and comparison. For floating-point calculations, external tools
like ``bc`` or ``awk`` are required since Bash only handles integers natively.

.. code-block:: bash

    ((a = 5 + 3))          # arithmetic assignment
    ((a++))                # increment
    ((a += 10))            # add and assign

    result=$((5 * 3))      # arithmetic expansion
    echo $((2 ** 10))      # 1024 (exponentiation)

Loops
-----

Bash provides multiple loop constructs for iteration over sequences, arrays, and
conditions. The ``for`` loop iterates over word lists, while ``while`` and ``until``
loops continue based on condition evaluation. The ``:`` builtin (equivalent to ``true``)
enables infinite loops commonly used for daemons and interactive menus.

.. code-block:: bash

    # for loop
    for i in {1..5}; do echo $i; done

    # C-style for
    for ((i=0; i<5; i++)); do echo $i; done

    # while loop
    while [[ $i -lt 5 ]]; do ((i++)); done

    # infinite loop
    while :; do sleep 1; done

    # iterate array
    for item in "${arr[@]}"; do echo "$item"; done

    # iterate with index
    for i in "${!arr[@]}"; do echo "$i: ${arr[$i]}"; done

Functions
---------

Functions encapsulate reusable logic with local variable scope support. Arguments are
accessed via positional parameters (``$1``, ``$2``, etc.), and return values are limited
to exit status codes (0-255). For returning strings or complex data, use command
substitution to capture function output or modify global variables.

.. code-block:: bash

    myfunc() {
      local var="local"    # local variable
      echo "arg1: $1"
      return 0             # return status
    }

    myfunc "hello"
    echo $?                # check return status

Here Documents & Strings
------------------------

Here documents embed multi-line text directly within scripts, while here strings pass
single-line input to commands expecting stdin. Both constructs avoid the need for
temporary files or complex ``echo`` pipelines. Quoting the delimiter (``'EOF'``) prevents
variable expansion within the here document.

.. code-block:: bash

    # here document
    cat <<EOF
    multi-line
    content here
    EOF

    # here document (no variable expansion)
    cat <<'EOF'
    $var not expanded
    EOF

    # here string
    bc <<< "1 + 2 * 3"

Read Input
----------

The ``read`` builtin captures user input with options for custom prompts, timeouts,
silent mode for passwords, and character limits. Setting ``IFS`` controls field splitting,
and the ``-r`` flag prevents backslash interpretation for raw input handling.

.. code-block:: bash

    read -p "Enter name: " name
    read -s -p "Password: " pass    # silent input
    read -t 5 -p "Quick: " ans      # 5 second timeout
    read -n 1 -p "Press key: " key  # single character

Read File Line by Line
----------------------

Processing files line-by-line with ``while read`` preserves whitespace and handles
special characters correctly when ``IFS`` is properly configured. Setting ``IFS=`` before
``read`` prevents leading/trailing whitespace trimming, and ``-r`` prevents backslash
escape interpretation.

.. code-block:: bash

    while IFS= read -r line; do
      echo "$line"
    done < file.txt

    # read with field separator
    while IFS=: read -r user _ uid gid _ home shell; do
      echo "$user: $home"
    done < /etc/passwd

Process Substitution
--------------------

Process substitution ``<()`` treats command output as a file descriptor, enabling
commands that require file arguments to accept dynamic input. This is particularly
useful for comparing outputs of two commands or feeding generated data to programs
that only accept file inputs.

.. code-block:: bash

    # compare two command outputs
    diff <(ls dir1) <(ls dir2)

    # feed command output as file
    while read line; do echo "$line"; done < <(ls -la)

Command Substitution
--------------------

Command substitution captures command output into variables for further processing. The
``$()`` syntax is preferred over legacy backticks for improved readability, easier
nesting, and consistent quoting behavior. The captured output has trailing newlines
stripped automatically.

.. code-block:: bash

    now=$(date +%Y-%m-%d)
    files=$(ls *.txt)
    count=$(wc -l < file.txt)

Check Command Exists
--------------------

Verifying command availability before execution prevents runtime errors and enables
graceful fallbacks in portable scripts. The ``command -v`` builtin is POSIX-compliant
and preferred over ``which``, which may behave inconsistently across systems.

.. code-block:: bash

    if command -v git &>/dev/null; then
      echo "git is installed"
    fi

    # alternative
    type -P git &>/dev/null && echo "found"

Error Handling
--------------

Robust scripts require explicit error handling to fail fast and provide meaningful
diagnostics. The ``set`` builtin configures shell behavior for error detection, while
``trap`` enables cleanup routines on exit, error, or signal reception. The combination
``set -euo pipefail`` is considered best practice for production scripts.

.. code-block:: bash

    set -e          # exit on error
    set -u          # error on undefined variable
    set -o pipefail # pipeline fails if any command fails
    set -x          # debug mode (print commands)

    # common combination
    set -euo pipefail

    # trap errors
    trap 'echo "Error at line $LINENO"' ERR
    trap 'cleanup' EXIT

Argument Parsing
----------------

Command-line argument parsing enables flexible script interfaces with options and
positional parameters. The ``getopts`` builtin handles POSIX-style short options with
automatic error handling, while manual parsing loops provide support for GNU-style long
options and more complex argument structures.

.. code-block:: bash

    usage() { echo "Usage: $0 [-h] [-f file] args..."; exit 1; }

    while getopts ":hf:" opt; do
      case $opt in
        h) usage ;;
        f) file="$OPTARG" ;;
        \?) echo "Invalid: -$OPTARG" >&2; exit 1 ;;
        :) echo "Option -$OPTARG requires argument" >&2; exit 1 ;;
      esac
    done
    shift $((OPTIND-1))

Long options with manual parsing:

.. code-block:: bash

    while [[ $# -gt 0 ]]; do
      case "$1" in
        -h|--help) usage; exit 0 ;;
        -f|--file) file="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) args+=("$1"); shift ;;
      esac
    done

Logging
-------

Structured logging with timestamps and severity levels improves script debugging and
operational monitoring. ANSI color codes enhance terminal readability by visually
distinguishing log levels. Redirecting warnings and errors to stderr (``>&2``) ensures
proper stream separation for pipeline compatibility.

.. code-block:: bash

    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[0;33m'
    readonly NC='\033[0m'

    log()   { echo -e "[$(date -Iseconds)] $*"; }
    info()  { echo -e "[$(date -Iseconds)] ${GREEN}INFO${NC} $*"; }
    warn()  { echo -e "[$(date -Iseconds)] ${YELLOW}WARN${NC} $*" >&2; }
    error() { echo -e "[$(date -Iseconds)] ${RED}ERROR${NC} $*" >&2; }

Regular Expressions
-------------------

Bash supports extended regular expression matching with the ``=~`` operator inside
``[[ ]]`` conditionals. When a match occurs, captured groups are stored in the
``BASH_REMATCH`` array, with index 0 containing the full match and subsequent indices
holding parenthesized subexpressions.

Bash regex matching with ``=~``:

.. code-block:: bash

    if [[ "$str" =~ ^[0-9]+$ ]]; then
      echo "numeric"
    fi

    # capture groups
    if [[ "hello123" =~ ([a-z]+)([0-9]+) ]]; then
      echo "${BASH_REMATCH[0]}"  # hello123 (full match)
      echo "${BASH_REMATCH[1]}"  # hello
      echo "${BASH_REMATCH[2]}"  # 123
    fi

grep Basic vs Extended Regex
----------------------------

The ``grep`` command supports basic regular expressions (BRE) by default and extended
regular expressions (ERE) with the ``-E`` flag. In BRE, metacharacters like ``?``, ``+``,
``{}``, and ``|`` require backslash escaping, while ERE treats them as special by default.
Perl-compatible regex (``-P``) offers additional features like lookahead and lookbehind.

.. code-block:: bash

    # basic regex (escape special chars)
    echo "987-123-4567" | grep "^[0-9]\{3\}-[0-9]\{3\}-[0-9]\{4\}$"

    # extended regex (no escaping)
    echo "987-123-4567" | grep -E "^[0-9]{3}-[0-9]{3}-[0-9]{4}$"

    # Perl regex
    echo "test@email.com" | grep -P "[\w.]+@[\w.]+"

Text Processing
---------------

Unix text processing tools form a powerful pipeline ecosystem for data transformation
and analysis. These utilities follow the Unix philosophy of doing one thing well and
composing via pipes. Mastering ``tr``, ``sort``, ``uniq``, ``cut``, ``awk``, and ``sed``
enables efficient manipulation of structured and unstructured text data.

.. code-block:: bash

    # tr - translate characters
    echo "hello" | tr 'a-z' 'A-Z'       # HELLO
    echo "a  b  c" | tr -s ' '          # squeeze spaces
    echo "a:b:c" | tr ':' '\n'          # replace : with newline

    # sort
    sort file.txt                       # sort lines
    sort -n file.txt                    # numeric sort
    sort -k2 file.txt                   # sort by 2nd field
    sort -u file.txt                    # unique sort
    sort -r file.txt                    # reverse sort

    # uniq (requires sorted input)
    sort file.txt | uniq                # remove duplicates
    sort file.txt | uniq -c             # count occurrences
    sort file.txt | uniq -d             # show only duplicates

    # cut
    cut -d: -f1 /etc/passwd             # first field
    cut -c1-10 file.txt                 # first 10 chars

    # awk
    awk '{print $1}' file.txt           # first column
    awk -F: '{print $1}' /etc/passwd    # with delimiter
    awk 'NR==5' file.txt                # 5th line

    # sed
    sed 's/old/new/' file.txt           # replace first
    sed 's/old/new/g' file.txt          # replace all
    sed -i 's/old/new/g' file.txt       # in-place edit
    sed -n '5p' file.txt                # print 5th line
    sed '5d' file.txt                   # delete 5th line

Temporary Files
---------------

The ``mktemp`` command creates secure temporary files and directories with unique,
unpredictable names. This prevents race conditions and symlink attacks that can occur
with predictable temporary filenames. Always combine with ``trap`` to ensure cleanup
on script exit, even when errors occur.

.. code-block:: bash

    tmpfile=$(mktemp)
    tmpdir=$(mktemp -d)

    trap 'rm -f "$tmpfile"' EXIT

    echo "data" > "$tmpfile"

Parallel Execution
------------------

Parallel execution accelerates batch operations by utilizing multiple CPU cores
simultaneously. Background jobs with ``&`` provide basic parallelism, ``xargs -P``
offers controlled concurrency for command-line tools, and GNU ``parallel`` provides
advanced features like progress bars, job logging, and remote execution.

.. code-block:: bash

    # background jobs
    cmd1 & cmd2 & wait

    # xargs parallel
    cat urls.txt | xargs -P 4 -I {} curl {}

    # GNU parallel
    parallel -j 4 gzip ::: *.txt

Script Template
---------------

A well-structured script template incorporates strict error handling, flexible argument
parsing, and consistent coding conventions. Starting from a proven template reduces
boilerplate and ensures best practices are followed from the beginning. This pattern
serves as a foundation for production-quality scripts.

.. code-block:: bash

    #!/usr/bin/env bash
    set -euo pipefail

    readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    readonly SCRIPT_NAME="$(basename "$0")"

    usage() {
      cat <<EOF
    Usage: $SCRIPT_NAME [OPTIONS] <args>

    Options:
      -h, --help     Show this help
      -v, --verbose  Verbose output
    EOF
    }

    main() {
      local verbose=false

      while [[ $# -gt 0 ]]; do
        case "$1" in
          -h|--help) usage; exit 0 ;;
          -v|--verbose) verbose=true; shift ;;
          --) shift; break ;;
          -*) echo "Unknown option: $1" >&2; exit 1 ;;
          *) break ;;
        esac
      done

      [[ $# -eq 0 ]] && { usage; exit 1; }

      # main logic here
    }

    main "$@"
