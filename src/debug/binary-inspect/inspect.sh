#!/usr/bin/env bash
# Binary inspection tools cheatsheet — run after building binary-inspect_test
# Works on both Linux (ELF) and macOS (Mach-O)
set -uo pipefail

BINARY="${1:?Usage: $0 <path-to-binary-inspect_test>}"

if [ ! -f "$BINARY" ]; then
  echo "ERROR: $BINARY not found. Build it first." >&2
  exit 1
fi

pass=0
fail=0

run() {
  local label="$1"; shift
  echo "──────────────────────────────────────────────────────"
  echo "[$label] $ $*"
  echo "──────────────────────────────────────────────────────"
  if "$@" 2>&1 | head -30; then
    ((pass++))
  else
    echo "(skipped or failed)"
    ((fail++))
  fi
  echo
}

has() { command -v "$1" &>/dev/null; }

IS_LINUX=false
if [[ "$(uname)" == "Linux" ]]; then
  IS_LINUX=true
fi

# --- file: works everywhere ---
echo "=== file ==="
run "file" file "$BINARY"

# --- nm: works everywhere ---
echo "=== nm ==="
run "nm -C" sh -c "nm -C '$BINARY' | head -20"
run "nm undefined" sh -c "nm -u '$BINARY' | head -10"
run "nm grep" sh -c "nm -C '$BINARY' | grep -E 'add|global|Widget' || true"

# --- strings: works everywhere ---
echo "=== strings ==="
run "strings grep" sh -c "strings '$BINARY' | grep 'binary-inspect' || true"
run "strings -n 10" sh -c "strings -n 10 '$BINARY' | head -10"

# --- size: works everywhere ---
echo "=== size ==="
run "size" size "$BINARY"

# --- c++filt: works everywhere ---
echo "=== c++filt ==="
run "c++filt" sh -c "nm '$BINARY' | grep Widget | c++filt || true"

# --- strip on a copy ---
echo "=== strip ==="
TMP=$(mktemp)
cp "$BINARY" "$TMP"
run "strip" strip "$TMP"
run "file stripped" file "$TMP"
run "nm stripped" sh -c "nm '$TMP' || true"
rm -f "$TMP"

if $IS_LINUX; then
  # --- Linux ELF tools ---
  echo "=== readelf ==="
  run "readelf -h" readelf -h "$BINARY"
  run "readelf -S" readelf -S "$BINARY"
  run "readelf -s" sh -c "readelf -s '$BINARY' | c++filt | head -20"
  run "readelf -l" sh -c "readelf -l '$BINARY' | head -20"
  run "readelf -d" sh -c "readelf -d '$BINARY' | head -20"
  run "readelf .rodata" sh -c "readelf -p .rodata '$BINARY' | head -20"

  echo "=== objdump ==="
  run "objdump disasm" sh -c "objdump -d -C '$BINARY' | grep -A5 '<add' | head -20"
  run "objdump .rodata" sh -c "objdump -s -j .rodata '$BINARY' | head -20"
  run "objdump symbols" sh -c "objdump -t -C '$BINARY' | grep -E 'add|global|Widget' || true"

  echo "=== ldd ==="
  run "ldd" ldd "$BINARY"
else
  # --- macOS Mach-O tools ---
  echo "=== otool (macOS equivalent of readelf/objdump/ldd) ==="
  run "otool -h (header)" otool -h "$BINARY"
  run "otool -l (load cmds)" sh -c "otool -l '$BINARY' | head -40"
  run "otool -L (dylibs)" otool -L "$BINARY"
  run "otool -tV (disasm)" sh -c "otool -tV '$BINARY' | head -30"

  if has objdump; then
    echo "=== objdump (llvm) ==="
    run "objdump --disassemble" sh -c "objdump --disassemble -C '$BINARY' | head -30"
  fi
fi

echo "========================================"
echo "PASS: $pass  FAIL: $fail"
echo "========================================"
[ "$fail" -eq 0 ]
