#!/usr/bin/env bash
#
# Hardware Information - Interactive Demo Script
# Run: ./hardware.sh [section]
# Note: Some commands require root privileges or Linux-specific tools
#
set -euo pipefail

readonly NC='\033[0m'
readonly BOLD='\033[1m'
readonly GREEN='\033[0;32m'
readonly CYAN='\033[0;36m'
readonly YELLOW='\033[0;33m'
readonly RED='\033[0;31m'

section() { echo -e "\n${BOLD}${GREEN}=== $1 ===${NC}\n"; }
code() { echo -e "${CYAN}$ $1${NC}"; }
output() { echo -e "${YELLOW}$1${NC}"; }
warn() { echo -e "${RED}(requires: $1)${NC}"; }

check_cmd() { command -v "$1" &>/dev/null; }
is_linux() { [[ "$OSTYPE" == "linux"* ]]; }
is_macos() { [[ "$OSTYPE" == "darwin"* ]]; }

# -----------------------------------------------------------------------------
demo_cpu() {
  section "CPU Information"

  if check_cmd nproc; then
    code 'nproc'
    output "$(nproc)"
  fi

  if is_macos; then
    code 'sysctl -n hw.logicalcpu'
    output "$(sysctl -n hw.logicalcpu)"

    code 'sysctl -n hw.physicalcpu'
    output "$(sysctl -n hw.physicalcpu)"

    code 'sysctl -n machdep.cpu.brand_string'
    output "$(sysctl -n machdep.cpu.brand_string)"
  fi

  if check_cmd lscpu; then
    code 'lscpu | head -15'
    lscpu | head -15
  fi

  if is_linux && [[ -f /proc/cpuinfo ]]; then
    code 'grep "model name" /proc/cpuinfo | head -1'
    grep "model name" /proc/cpuinfo | head -1
    echo

    code 'grep -c ^processor /proc/cpuinfo'
    output "$(grep -c ^processor /proc/cpuinfo) logical CPUs"
  fi
}

# -----------------------------------------------------------------------------
demo_memory() {
  section "Memory Information"

  if check_cmd free; then
    code 'free -h'
    free -h
    echo
  fi

  if is_macos; then
    code 'sysctl -n hw.memsize | awk "{print \$1/1024/1024/1024 \" GB\"}"'
    sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'
    echo

    code 'vm_stat | head -10'
    vm_stat | head -10
  fi

  if is_linux && [[ -f /proc/meminfo ]]; then
    code 'head -5 /proc/meminfo'
    head -5 /proc/meminfo
  fi
}

# -----------------------------------------------------------------------------
demo_disk() {
  section "Disk and Storage"

  code 'df -h'
  df -h
  echo

  code 'df -h / | tail -1'
  df -h / | tail -1
  echo

  code 'du -sh ~/* 2>/dev/null | sort -h | tail -5'
  du -sh ~/* 2>/dev/null | sort -h | tail -5 || echo "(permission denied on some dirs)"
  echo

  if check_cmd lsblk; then
    code 'lsblk'
    lsblk
  elif is_macos; then
    code 'diskutil list | head -20'
    diskutil list | head -20
  fi
}

# -----------------------------------------------------------------------------
demo_pci() {
  section "PCI Devices"

  if ! check_cmd lspci; then
    if is_macos; then
      code 'system_profiler SPPCIDataType | head -30'
      system_profiler SPPCIDataType 2>/dev/null | head -30 || echo "(no PCI data)"
    else
      warn "lspci (install with: apt install pciutils)"
    fi
    return
  fi

  code 'lspci | head -15'
  lspci | head -15
  echo

  code 'lspci | grep -i vga'
  lspci | grep -i vga || echo "(no VGA device found)"
  echo

  code 'lspci | grep -i network'
  lspci | grep -i network || echo "(no network device found)"
  echo

  code 'lspci -nn | head -10'
  lspci -nn | head -10
}

# -----------------------------------------------------------------------------
demo_summary() {
  section "System Summary"

  code 'uname -a'
  output "$(uname -a)"

  if is_macos; then
    code 'system_profiler SPHardwareDataType | grep -E "Model|Processor|Memory|Serial"'
    system_profiler SPHardwareDataType | grep -E "Model|Processor|Memory|Serial"
  fi

  if check_cmd hostnamectl; then
    code 'hostnamectl'
    hostnamectl
  fi

  if check_cmd lshw; then
    echo
    echo "lshw available - run with: sudo lshw -short"
  fi

  if check_cmd dmidecode; then
    echo
    echo "dmidecode available - run with: sudo dmidecode -t system"
  fi
}

# -----------------------------------------------------------------------------
run_all() {
  demo_cpu
  demo_memory
  demo_disk
  demo_pci
  demo_summary
}

# -----------------------------------------------------------------------------
usage() {
  cat <<EOF
Hardware Information - Interactive Demo

Usage: $0 [section]

Sections:
  cpu         CPU information
  memory      Memory information
  disk        Disk and storage
  pci         PCI devices
  summary     System summary
  all         Run all demos (default)

Note: Some commands require root privileges or Linux-specific tools.
For GPU information, see gpu.sh

EOF
}

# Main
case "${1:-all}" in
  cpu)     demo_cpu ;;
  memory)  demo_memory ;;
  disk)    demo_disk ;;
  pci)     demo_pci ;;
  gpu)     demo_gpu ;;
  summary) demo_summary ;;
  all)     run_all ;;
  -h|--help) usage ;;
  *) echo "Unknown section: $1"; usage; exit 1 ;;
esac
