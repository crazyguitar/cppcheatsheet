#!/usr/bin/env bash
#
# GPU Information - Interactive Demo Script
# Run: ./gpu.sh [section]
# Note: Requires NVIDIA GPU and nvidia-smi
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
warn() { echo -e "${RED}$1${NC}"; }

check_cmd() { command -v "$1" &>/dev/null; }

# -----------------------------------------------------------------------------
demo_basic() {
  section "Basic GPU Information"

  if check_cmd lspci; then
    code 'lspci | grep -i vga'
    lspci | grep -i vga || echo "(no VGA device)"
    echo

    code 'lspci | grep -i nvidia'
    lspci | grep -i nvidia || echo "(no NVIDIA device)"
  fi

  if check_cmd nvidia-smi; then
    echo
    code 'nvidia-smi -L'
    nvidia-smi -L
  else
    warn "nvidia-smi not found - NVIDIA driver not installed"
  fi
}

# -----------------------------------------------------------------------------
demo_status() {
  section "GPU Status"

  if ! check_cmd nvidia-smi; then
    warn "nvidia-smi not found"
    return
  fi

  code 'nvidia-smi'
  nvidia-smi
}

# -----------------------------------------------------------------------------
demo_query() {
  section "Query Format Examples"

  if ! check_cmd nvidia-smi; then
    warn "nvidia-smi not found"
    echo
    echo "Example query commands:"
    code 'nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv'
    code 'nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv'
    return
  fi

  code 'nvidia-smi --query-gpu=name,driver_version --format=csv'
  nvidia-smi --query-gpu=name,driver_version --format=csv
  echo

  code 'nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv'
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
  echo

  code 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv'
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv
  echo

  code 'nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv'
  nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv
  echo

  code 'nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv'
  nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv
}

# -----------------------------------------------------------------------------
demo_processes() {
  section "GPU Processes"

  if ! check_cmd nvidia-smi; then
    warn "nvidia-smi not found"
    return
  fi

  code 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "No compute processes running"
}

# -----------------------------------------------------------------------------
demo_monitoring() {
  section "Monitoring Commands"

  if ! check_cmd nvidia-smi; then
    warn "nvidia-smi not found"
    echo
    echo "Example monitoring commands:"
    code 'nvidia-smi -l 1                # Update every second'
    code 'nvidia-smi dmon -s u           # Utilization monitoring'
    code 'nvidia-smi pmon -s m           # Process memory monitoring'
    return
  fi

  echo "Monitoring commands (Ctrl+C to stop):"
  code 'nvidia-smi -l 1                  # Update every second'
  code 'nvidia-smi dmon -s u             # Device utilization'
  code 'nvidia-smi dmon -s p             # Power monitoring'
  code 'nvidia-smi dmon -s t             # Temperature monitoring'
  code 'nvidia-smi pmon -s u             # Process utilization'
  echo
  code 'watch -n 1 nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv'

  echo
  echo "Running dmon for 3 seconds..."
  timeout 3 nvidia-smi dmon -s u 2>/dev/null || true
}

# -----------------------------------------------------------------------------
demo_topology() {
  section "Multi-GPU Topology"

  if ! check_cmd nvidia-smi; then
    warn "nvidia-smi not found"
    return
  fi

  code 'nvidia-smi topo -m'
  nvidia-smi topo -m 2>/dev/null || echo "(topology not available)"
}

# -----------------------------------------------------------------------------
demo_cuda() {
  section "CUDA Environment"

  if check_cmd nvcc; then
    code 'nvcc --version'
    nvcc --version
  else
    echo "nvcc not found - CUDA toolkit not installed"
  fi

  echo
  echo "CUDA environment variables:"
  code 'export CUDA_VISIBLE_DEVICES=0       # Use only GPU 0'
  code 'export CUDA_VISIBLE_DEVICES=0,1     # Use GPUs 0 and 1'
  code 'export CUDA_VISIBLE_DEVICES=""      # Hide all GPUs'

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo
    output "Current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi
}

# -----------------------------------------------------------------------------
demo_management() {
  section "Management Commands (require root)"

  echo "These commands typically require root privileges:"
  echo
  code 'nvidia-smi -pm 1                 # Enable persistence mode'
  code 'nvidia-smi -pl 250               # Set power limit to 250W'
  code 'nvidia-smi -lgc 1200,1800        # Lock graphics clock range'
  code 'nvidia-smi -rgc                  # Reset graphics clocks'
  code 'nvidia-smi -c 0                  # Set compute mode (default)'
  code 'nvidia-smi -r                    # Reset GPU'
}

# -----------------------------------------------------------------------------
run_all() {
  demo_basic
  demo_status
  demo_query
  demo_processes
  demo_monitoring
  demo_topology
  demo_cuda
  demo_management
}

# -----------------------------------------------------------------------------
usage() {
  cat <<EOF
GPU Information - Interactive Demo

Usage: $0 [section]

Sections:
  basic       Basic GPU information
  status      nvidia-smi status
  query       Query format examples
  processes   GPU processes
  monitor     Monitoring commands
  topology    Multi-GPU topology
  cuda        CUDA environment
  manage      Management commands
  all         Run all demos (default)

Note: Requires NVIDIA GPU and nvidia-smi for most features.

EOF
}

# Main
case "${1:-all}" in
  basic)    demo_basic ;;
  status)   demo_status ;;
  query)    demo_query ;;
  processes) demo_processes ;;
  monitor)  demo_monitoring ;;
  topology) demo_topology ;;
  cuda)     demo_cuda ;;
  manage)   demo_management ;;
  all)      run_all ;;
  -h|--help) usage ;;
  *) echo "Unknown section: $1"; usage; exit 1 ;;
esac
