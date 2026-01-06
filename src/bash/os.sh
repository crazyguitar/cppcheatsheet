#!/usr/bin/env bash
#
# Operating System Commands - Interactive Demo Script
# Run: ./os.sh [section]
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
run() { code "$1"; eval "$1" 2>/dev/null || true; echo; }

# -----------------------------------------------------------------------------
demo_date() {
  section "Date and Time"

  code 'date'
  output "$(date)"

  code 'date +"%Y-%m-%d"'
  output "$(date +"%Y-%m-%d")"

  code 'date +"%Y-%m-%d %H:%M:%S"'
  output "$(date +"%Y-%m-%d %H:%M:%S")"

  code 'date +%s  # Unix timestamp'
  output "$(date +%s)"

  code 'date -Iseconds  # ISO 8601'
  output "$(date -Iseconds 2>/dev/null || date +"%Y-%m-%dT%H:%M:%S%z")"

  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "\nmacOS date arithmetic:"
    code 'date -j -v-1d +"%Y-%m-%d"  # Yesterday'
    output "$(date -j -v-1d +"%Y-%m-%d")"
    code 'date -j -v+7d +"%Y-%m-%d"  # 7 days from now'
    output "$(date -j -v+7d +"%Y-%m-%d")"
  else
    echo -e "\nLinux date arithmetic:"
    code 'date +%Y-%m-%d -d "1 day ago"'
    output "$(date +%Y-%m-%d -d "1 day ago")"
    code 'date +%Y-%m-%d -d "7 days"'
    output "$(date +%Y-%m-%d -d "7 days")"
  fi
}

# -----------------------------------------------------------------------------
demo_find() {
  section "Find Command"

  local tmpdir
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' RETURN

  # Create test files
  mkdir -p "$tmpdir/subdir"
  echo "hello" > "$tmpdir/file1.txt"
  echo "world" > "$tmpdir/file2.txt"
  echo "test" > "$tmpdir/subdir/file3.log"
  touch -d "2 days ago" "$tmpdir/old.txt" 2>/dev/null || touch "$tmpdir/old.txt"

  echo "Created test directory: $tmpdir"
  run "ls -la $tmpdir"

  code "find $tmpdir -name '*.txt'"
  find "$tmpdir" -name "*.txt"
  echo

  code "find $tmpdir -type f"
  find "$tmpdir" -type f
  echo

  code "find $tmpdir -type d"
  find "$tmpdir" -type d
  echo

  code "find $tmpdir -name '*.txt' -exec basename {} \\;"
  find "$tmpdir" -name "*.txt" -exec basename {} \;
  echo
}

# -----------------------------------------------------------------------------
demo_sysinfo() {
  section "System Information"

  code 'uname -a'
  output "$(uname -a)"

  code 'uname -s  # Kernel name'
  output "$(uname -s)"

  code 'uname -r  # Kernel release'
  output "$(uname -r)"

  code 'uname -m  # Architecture'
  output "$(uname -m)"

  code 'hostname'
  output "$(hostname)"

  code 'uptime'
  output "$(uptime)"

  if [[ "$OSTYPE" == "darwin"* ]]; then
    code 'sw_vers'
    sw_vers
  elif [[ -f /etc/os-release ]]; then
    code 'cat /etc/os-release | head -5'
    head -5 /etc/os-release
  fi
}

# -----------------------------------------------------------------------------
demo_process() {
  section "Process Management"

  code 'ps aux | head -5'
  ps aux | head -5
  echo

  code 'pgrep -l bash | head -3'
  pgrep -l bash | head -3 || echo "(no results)"
  echo

  code 'echo $$  # Current PID'
  output "$$"

  if command -v pstree &>/dev/null; then
    code 'pstree -p $$ | head -3'
    pstree -p $$ 2>/dev/null | head -3 || echo "(pstree not available)"
  fi
}

# -----------------------------------------------------------------------------
demo_files() {
  section "File Operations"

  local tmpdir
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' RETURN

  echo "Working in: $tmpdir"

  code "mkdir -p $tmpdir/a/b/c"
  mkdir -p "$tmpdir/a/b/c"
  output "Created nested directories"

  code "echo 'hello world' > $tmpdir/test.txt"
  echo 'hello world' > "$tmpdir/test.txt"

  code "cat $tmpdir/test.txt"
  output "$(cat "$tmpdir/test.txt")"

  code "wc -l $tmpdir/test.txt"
  output "$(wc -l < "$tmpdir/test.txt") lines"

  code "ln -s $tmpdir/test.txt $tmpdir/link.txt"
  ln -s "$tmpdir/test.txt" "$tmpdir/link.txt"
  code "ls -la $tmpdir/*.txt"
  ls -la "$tmpdir"/*.txt
}

# -----------------------------------------------------------------------------
demo_compression() {
  section "Compression"

  local tmpdir
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' RETURN

  echo "test content" > "$tmpdir/file.txt"

  code "tar -cvf $tmpdir/archive.tar -C $tmpdir file.txt"
  tar -cvf "$tmpdir/archive.tar" -C "$tmpdir" file.txt
  echo

  code "tar -tzvf after gzip"
  gzip "$tmpdir/archive.tar"
  tar -tzvf "$tmpdir/archive.tar.gz"
  echo

  code "ls -lh $tmpdir/"
  ls -lh "$tmpdir/"
}

# -----------------------------------------------------------------------------
demo_env() {
  section "Environment and Shell"

  code 'echo $SHELL'
  output "$SHELL"

  code 'echo $HOME'
  output "$HOME"

  code 'echo $PATH | tr ":" "\n" | head -3'
  echo "$PATH" | tr ':' '\n' | head -3

  code 'which bash'
  output "$(which bash)"

  code 'type ls'
  type ls
}

# -----------------------------------------------------------------------------
run_all() {
  demo_date
  demo_find
  demo_sysinfo
  demo_process
  demo_files
  demo_compression
  demo_env
}

# -----------------------------------------------------------------------------
usage() {
  cat <<EOF
Operating System Commands - Interactive Demo

Usage: $0 [section]

Sections:
  date        Date and time commands
  find        Find command examples
  sysinfo     System information
  process     Process management
  files       File operations
  compress    Compression utilities
  env         Environment and shell
  all         Run all demos (default)

EOF
}

# Main
case "${1:-all}" in
  date)     demo_date ;;
  find)     demo_find ;;
  sysinfo)  demo_sysinfo ;;
  process)  demo_process ;;
  files)    demo_files ;;
  compress) demo_compression ;;
  env)      demo_env ;;
  all)      run_all ;;
  -h|--help) usage ;;
  *) echo "Unknown section: $1"; usage; exit 1 ;;
esac
