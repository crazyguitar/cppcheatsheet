#!/usr/bin/env bash
#
# Bash Basic Cheatsheet - Interactive Demo Script
# Run: ./basic.sh [section]
# Examples:
#   ./basic.sh              # Run all demos
#   ./basic.sh strings      # Run string manipulation demo
#   ./basic.sh arrays       # Run arrays demo
#
set -euo pipefail

readonly NC='\033[0m'
readonly BOLD='\033[1m'
readonly GREEN='\033[0;32m'
readonly CYAN='\033[0;36m'
readonly YELLOW='\033[0;33m'

section() { echo -e "\n${BOLD}${GREEN}=== $1 ===${NC}\n"; }
code() { echo -e "${CYAN}$ $1${NC}"; }
output() { echo -e "${YELLOW}$1${NC}"; }

# -----------------------------------------------------------------------------
demo_special_params() {
  section "Special Parameters"

  code 'echo "Script name: $0"'
  output "Script name: $0"

  code 'echo "Process ID: $$"'
  output "Process ID: $$"

  code 'echo "Last exit status: $?"'
  output "Last exit status: $?"

  inner_func() {
    code 'echo "All args (\$@): $@"'
    output "All args (\$@): $@"
    code 'echo "Arg count (\$#): $#"'
    output "Arg count (\$#): $#"
  }
  echo -e "\nCalling: inner_func \"hello\" \"world\" \"bash\""
  inner_func "hello" "world" "bash"
}

# -----------------------------------------------------------------------------
demo_brace_expansion() {
  section "Brace Expansion"

  code 'echo file.{txt,pdf,md}'
  output "$(echo file.{txt,pdf,md})"

  code 'echo {1..5}'
  output "$(echo {1..5})"

  code 'echo {a..e}'
  output "$(echo {a..e})"

  code 'echo {01..05}'
  output "$(echo {01..05})"

  code 'echo {1..10..2}'
  output "$(echo {1..10..2})"
}

# -----------------------------------------------------------------------------
demo_strings() {
  section "String Manipulation"

  str="Hello World"
  code "str=\"$str\""

  code 'echo "Length: ${#str}"'
  output "Length: ${#str}"

  code 'echo "Substring from 6: ${str:6}"'
  output "Substring from 6: ${str:6}"

  code 'echo "First 5 chars: ${str:0:5}"'
  output "First 5 chars: ${str:0:5}"

  code 'echo "Last 5 chars: ${str: -5}"'
  output "Last 5 chars: ${str: -5}"

  code 'echo "Uppercase: ${str^^}"'
  output "Uppercase: ${str^^}"

  code 'echo "Lowercase: ${str,,}"'
  output "Lowercase: ${str,,}"
}

# -----------------------------------------------------------------------------
demo_pattern_removal() {
  section "Pattern Removal"

  path="/home/user/documents/file.tar.gz"
  code "path=\"$path\""

  code 'echo "Remove shortest prefix */: ${path#*/}"'
  output "Remove shortest prefix */: ${path#*/}"

  code 'echo "Remove longest prefix */: ${path##*/}"'
  output "Remove longest prefix */: ${path##*/}"

  code 'echo "Remove shortest suffix .*: ${path%.*}"'
  output "Remove shortest suffix .*: ${path%.*}"

  code 'echo "Remove longest suffix .*: ${path%%.*}"'
  output "Remove longest suffix .*: ${path%%.*}"
}

# -----------------------------------------------------------------------------
demo_pattern_substitution() {
  section "Pattern Substitution"

  text="foo bar foo baz foo"
  code "text=\"$text\""

  code 'echo "Replace first foo: ${text/foo/XXX}"'
  output "Replace first foo: ${text/foo/XXX}"

  code 'echo "Replace all foo: ${text//foo/XXX}"'
  output "Replace all foo: ${text//foo/XXX}"

  code 'echo "Replace if starts with: ${text/#foo/XXX}"'
  output "Replace if starts with: ${text/#foo/XXX}"

  code 'echo "Replace if ends with: ${text/%foo/XXX}"'
  output "Replace if ends with: ${text/%foo/XXX}"
}

# -----------------------------------------------------------------------------
demo_default_values() {
  section "Variable Expansion with Defaults"

  unset myvar 2>/dev/null || true

  code 'echo "Default if unset: ${myvar:-default_value}"'
  output "Default if unset: ${myvar:-default_value}"

  code 'echo "myvar is still unset: ${myvar-unset}"'
  output "myvar is still unset: ${myvar-unset}"

  code 'echo "Assign if unset: ${myvar:=assigned_value}"'
  output "Assign if unset: ${myvar:=assigned_value}"

  code 'echo "myvar is now: $myvar"'
  output "myvar is now: $myvar"

  code 'echo "Use alt if set: ${myvar:+alternative}"'
  output "Use alt if set: ${myvar:+alternative}"
}

# -----------------------------------------------------------------------------
demo_arrays() {
  section "Arrays"

  code 'arr=(apple banana cherry date)'
  arr=(apple banana cherry date)

  code 'echo "First element: ${arr[0]}"'
  output "First element: ${arr[0]}"

  code 'echo "All elements: ${arr[@]}"'
  output "All elements: ${arr[@]}"

  code 'echo "Array length: ${#arr[@]}"'
  output "Array length: ${#arr[@]}"

  code 'echo "All indices: ${!arr[@]}"'
  output "All indices: ${!arr[@]}"

  code 'arr+=(elderberry fig)'
  arr+=(elderberry fig)
  code 'echo "After append: ${arr[@]}"'
  output "After append: ${arr[@]}"

  echo -e "\nIterating with index:"
  code 'for i in "${!arr[@]}"; do echo "  [$i] = ${arr[$i]}"; done'
  for i in "${!arr[@]}"; do
    output "  [$i] = ${arr[$i]}"
  done
}

# -----------------------------------------------------------------------------
demo_associative_arrays() {
  section "Associative Arrays (Dictionary)"

  code 'declare -A colors'
  declare -A colors

  code 'colors=([red]="#FF0000" [green]="#00FF00" [blue]="#0000FF")'
  colors=([red]="#FF0000" [green]="#00FF00" [blue]="#0000FF")

  code 'echo "Red: ${colors[red]}"'
  output "Red: ${colors[red]}"

  code 'echo "All keys: ${!colors[@]}"'
  output "All keys: ${!colors[@]}"

  code 'echo "All values: ${colors[@]}"'
  output "All values: ${colors[@]}"

  echo -e "\nIterating:"
  code 'for key in "${!colors[@]}"; do echo "  $key -> ${colors[$key]}"; done'
  for key in "${!colors[@]}"; do
    output "  $key -> ${colors[$key]}"
  done

  code '[[ -v colors[red] ]] && echo "red exists"'
  [[ -v colors[red] ]] && output "red exists"
}

# -----------------------------------------------------------------------------
demo_conditionals() {
  section "Conditionals"

  echo "File tests:"
  code '[[ -f /etc/passwd ]] && echo "/etc/passwd is a file"'
  [[ -f /etc/passwd ]] && output "/etc/passwd is a file"

  code '[[ -d /tmp ]] && echo "/tmp is a directory"'
  [[ -d /tmp ]] && output "/tmp is a directory"

  echo -e "\nString tests:"
  str="hello"
  code "str=\"$str\""

  code '[[ -n "$str" ]] && echo "str is non-empty"'
  [[ -n "$str" ]] && output "str is non-empty"

  code '[[ "$str" == "hello" ]] && echo "str equals hello"'
  [[ "$str" == "hello" ]] && output "str equals hello"

  echo -e "\nNumeric tests:"
  a=10 b=20
  code "a=$a b=$b"

  code '[[ $a -lt $b ]] && echo "$a < $b"'
  [[ $a -lt $b ]] && output "$a < $b"
}

# -----------------------------------------------------------------------------
demo_arithmetic() {
  section "Arithmetic"

  code '((x = 5 + 3)); echo "5 + 3 = $x"'
  ((x = 5 + 3)); output "5 + 3 = $x"

  code '((x++)); echo "After increment: $x"'
  ((x++)); output "After increment: $x"

  code 'echo "2^10 = $((2 ** 10))"'
  output "2^10 = $((2 ** 10))"

  code 'echo "17 % 5 = $((17 % 5))"'
  output "17 % 5 = $((17 % 5))"
}

# -----------------------------------------------------------------------------
demo_loops() {
  section "Loops"

  echo "For loop with range:"
  code 'for i in {1..5}; do echo -n "$i "; done'
  for i in {1..5}; do echo -n "$i "; done
  echo

  echo -e "\nC-style for loop:"
  code 'for ((i=0; i<3; i++)); do echo -n "$i "; done'
  for ((i=0; i<3; i++)); do echo -n "$i "; done
  echo

  echo -e "\nWhile loop:"
  code 'n=3; while ((n > 0)); do echo -n "$n "; ((n--)); done'
  n=3; while ((n > 0)); do echo -n "$n "; ((n--)); done
  echo
}

# -----------------------------------------------------------------------------
demo_functions() {
  section "Functions"

  greet() {
    local name="${1:-World}"
    echo "Hello, $name!"
    return 0
  }

  code 'greet() { local name="${1:-World}"; echo "Hello, $name!"; }'

  code 'greet'
  output "$(greet)"

  code 'greet "Bash"'
  output "$(greet "Bash")"

  add() {
    echo $(($1 + $2))
  }

  code 'add() { echo $(($1 + $2)); }'
  code 'result=$(add 10 20); echo "10 + 20 = $result"'
  result=$(add 10 20)
  output "10 + 20 = $result"
}

# -----------------------------------------------------------------------------
demo_regex() {
  section "Regular Expressions"

  str="hello123world"
  code "str=\"$str\""

  code '[[ "$str" =~ ^[a-z]+[0-9]+[a-z]+$ ]] && echo "Pattern matched"'
  [[ "$str" =~ ^[a-z]+[0-9]+[a-z]+$ ]] && output "Pattern matched"

  code '[[ "$str" =~ ([a-z]+)([0-9]+)([a-z]+) ]]'
  if [[ "$str" =~ ([a-z]+)([0-9]+)([a-z]+) ]]; then
    code 'echo "Full match: ${BASH_REMATCH[0]}"'
    output "Full match: ${BASH_REMATCH[0]}"
    code 'echo "Group 1: ${BASH_REMATCH[1]}"'
    output "Group 1: ${BASH_REMATCH[1]}"
    code 'echo "Group 2: ${BASH_REMATCH[2]}"'
    output "Group 2: ${BASH_REMATCH[2]}"
    code 'echo "Group 3: ${BASH_REMATCH[3]}"'
    output "Group 3: ${BASH_REMATCH[3]}"
  fi
}

# -----------------------------------------------------------------------------
demo_text_processing() {
  section "Text Processing"

  echo "tr - translate characters:"
  code 'echo "hello" | tr "a-z" "A-Z"'
  output "$(echo "hello" | tr "a-z" "A-Z")"

  code 'echo "a:b:c" | tr ":" "\n"'
  echo "a:b:c" | tr ":" "\n" | while read -r line; do output "$line"; done

  echo -e "\nsort and uniq:"
  code 'echo "c a b a c b" | tr " " "\n" | sort | uniq -c'
  echo "c a b a c b" | tr " " "\n" | sort | uniq -c | while read -r line; do output "$line"; done

  echo -e "\ncut:"
  code 'echo "one:two:three" | cut -d: -f2'
  output "$(echo "one:two:three" | cut -d: -f2)"
}

# -----------------------------------------------------------------------------
demo_here_docs() {
  section "Here Documents & Strings"

  echo "Here document:"
  code 'cat <<EOF'
  cat <<EOF
  Line 1
  Line 2
  Variable: $USER
EOF

  echo -e "\nHere string:"
  code 'bc <<< "10 * 5"'
  output "$(bc <<< "10 * 5")"

  code 'read -r first rest <<< "hello world bash"'
  read -r first rest <<< "hello world bash"
  code 'echo "first=$first rest=$rest"'
  output "first=$first rest=$rest"
}

# -----------------------------------------------------------------------------
demo_error_handling() {
  section "Error Handling"

  code 'set -euo pipefail  # Exit on error, undefined var, pipe failure'
  echo "(Already enabled in this script)"

  echo -e "\nTrap example:"
  code 'trap "echo Cleanup on exit" EXIT'

  cleanup_demo() {
    local tmpfile
    tmpfile=$(mktemp)
    trap 'rm -f "$tmpfile"; echo "Cleaned up $tmpfile"' RETURN
    echo "Created: $tmpfile"
    echo "data" > "$tmpfile"
  }

  code 'cleanup_demo  # Creates temp file, cleans up on return'
  cleanup_demo
}

# -----------------------------------------------------------------------------
run_all() {
  demo_special_params
  demo_brace_expansion
  demo_strings
  demo_pattern_removal
  demo_pattern_substitution
  demo_default_values
  demo_arrays
  demo_associative_arrays
  demo_conditionals
  demo_arithmetic
  demo_loops
  demo_functions
  demo_regex
  demo_text_processing
  demo_here_docs
  demo_error_handling
}

# -----------------------------------------------------------------------------
usage() {
  cat <<EOF
Bash Basic Cheatsheet - Interactive Demo

Usage: $0 [section]

Sections:
  params        Special parameters
  brace         Brace expansion
  strings       String manipulation
  removal       Pattern removal
  subst         Pattern substitution
  defaults      Variable defaults
  arrays        Indexed arrays
  dict          Associative arrays
  cond          Conditionals
  math          Arithmetic
  loops         Loops
  func          Functions
  regex         Regular expressions
  text          Text processing
  heredoc       Here documents
  error         Error handling
  all           Run all demos (default)

EOF
}

# Main
case "${1:-all}" in
  params)   demo_special_params ;;
  brace)    demo_brace_expansion ;;
  strings)  demo_strings ;;
  removal)  demo_pattern_removal ;;
  subst)    demo_pattern_substitution ;;
  defaults) demo_default_values ;;
  arrays)   demo_arrays ;;
  dict)     demo_associative_arrays ;;
  cond)     demo_conditionals ;;
  math)     demo_arithmetic ;;
  loops)    demo_loops ;;
  func)     demo_functions ;;
  regex)    demo_regex ;;
  text)     demo_text_processing ;;
  heredoc)  demo_here_docs ;;
  error)    demo_error_handling ;;
  all)      run_all ;;
  -h|--help) usage ;;
  *) echo "Unknown section: $1"; usage; exit 1 ;;
esac
