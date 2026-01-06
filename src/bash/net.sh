#!/usr/bin/env bash
#
# Network Commands - Interactive Demo Script
# Run: ./net.sh [section]
# Note: Some commands require root privileges or specific tools installed
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

# -----------------------------------------------------------------------------
demo_config() {
  section "Network Configuration"

  if check_cmd ip; then
    code 'ip addr show | head -20'
    ip addr show | head -20
    echo

    code 'ip route'
    ip route
    echo
  elif check_cmd ifconfig; then
    code 'ifconfig | head -20'
    ifconfig | head -20
    echo
  fi

  if [[ "$OSTYPE" == "darwin"* ]]; then
    code 'networksetup -listallhardwareports | head -10'
    networksetup -listallhardwareports | head -10 || true
  fi
}

# -----------------------------------------------------------------------------
demo_connections() {
  section "Connection Information"

  if check_cmd ss; then
    code 'ss -tuln | head -10'
    ss -tuln | head -10
    echo
  fi

  code 'netstat -an | head -15'
  if [[ "$OSTYPE" == "darwin"* ]]; then
    netstat -an | head -15
  else
    netstat -an 2>/dev/null | head -15 || echo "(netstat not available)"
  fi
  echo

  if check_cmd lsof; then
    code 'lsof -i -P | head -10'
    lsof -i -P 2>/dev/null | head -10 || echo "(requires privileges)"
  fi
}

# -----------------------------------------------------------------------------
demo_connectivity() {
  section "Connectivity Testing"

  code 'ping -c 3 8.8.8.8'
  ping -c 3 8.8.8.8 2>/dev/null || echo "(ping failed or blocked)"
  echo

  if check_cmd nc; then
    code 'nc -zv -w 3 google.com 80'
    nc -zv -w 3 google.com 80 2>&1 || true
    echo

    code 'nc -zv -w 3 google.com 443'
    nc -zv -w 3 google.com 443 2>&1 || true
  fi
}

# -----------------------------------------------------------------------------
demo_dns() {
  section "DNS Lookup"

  if check_cmd dig; then
    code 'dig +short google.com'
    dig +short google.com
    echo

    code 'dig +short google.com MX'
    dig +short google.com MX
    echo

    code 'dig +short -x 8.8.8.8'
    dig +short -x 8.8.8.8
  elif check_cmd nslookup; then
    code 'nslookup google.com'
    nslookup google.com | head -10
  elif check_cmd host; then
    code 'host google.com'
    host google.com
  else
    warn "dig, nslookup, or host"
  fi
}

# -----------------------------------------------------------------------------
demo_curl() {
  section "Data Transfer - curl"

  if ! check_cmd curl; then
    warn "curl"
    return
  fi

  code 'curl -I -s https://httpbin.org/get | head -10'
  curl -I -s https://httpbin.org/get | head -10
  echo

  code 'curl -s https://httpbin.org/ip'
  curl -s https://httpbin.org/ip
  echo

  code 'curl -s -X POST -d "name=test" https://httpbin.org/post | head -15'
  curl -s -X POST -d "name=test" https://httpbin.org/post | head -15
}

# -----------------------------------------------------------------------------
demo_nmap() {
  section "Port Scanning - nmap"

  if ! check_cmd nmap; then
    warn "nmap (install with: brew install nmap / apt install nmap)"
    echo
    echo "Example commands:"
    code 'nmap -sn 192.168.1.0/24        # Ping scan'
    code 'nmap -p 22,80,443 host         # Specific ports'
    code 'nmap -sV host                  # Service detection'
    code 'nmap -A host                   # Aggressive scan'
    return
  fi

  code 'nmap -sn 127.0.0.1'
  nmap -sn 127.0.0.1
  echo

  code 'nmap -p 22,80,443 127.0.0.1'
  nmap -p 22,80,443 127.0.0.1 2>/dev/null || echo "(scan completed)"
}

# -----------------------------------------------------------------------------
demo_tcpdump() {
  section "Packet Capture - tcpdump"

  if ! check_cmd tcpdump; then
    warn "tcpdump"
    echo
    echo "Example commands (require root):"
    code 'tcpdump -i any -c 10           # Capture 10 packets'
    code 'tcpdump -i eth0 port 80        # HTTP traffic'
    code 'tcpdump -i any host 8.8.8.8    # Traffic to/from host'
    code 'tcpdump -w capture.pcap        # Write to file'
    return
  fi

  echo "tcpdump requires root privileges. Example commands:"
  code 'sudo tcpdump -i any -c 5'
  code 'sudo tcpdump -i any port 80 -c 10'
  code 'sudo tcpdump -i any host 8.8.8.8 -c 5'
  code 'sudo tcpdump -nn -i any -c 10'
}

# -----------------------------------------------------------------------------
demo_bandwidth() {
  section "Network Bandwidth"

  if check_cmd iftop; then
    echo "iftop available - run with: sudo iftop"
  fi

  if check_cmd nethogs; then
    echo "nethogs available - run with: sudo nethogs"
  fi

  if check_cmd iperf3; then
    echo "iperf3 available for bandwidth testing:"
    code 'iperf3 -s                      # Start server'
    code 'iperf3 -c server_ip            # Run client test'
  else
    warn "iperf3 (install with: brew install iperf3 / apt install iperf3)"
  fi
}

# -----------------------------------------------------------------------------
run_all() {
  demo_config
  demo_connections
  demo_connectivity
  demo_dns
  demo_curl
  demo_nmap
  demo_tcpdump
  demo_bandwidth
}

# -----------------------------------------------------------------------------
usage() {
  cat <<EOF
Network Commands - Interactive Demo

Usage: $0 [section]

Sections:
  config      Network configuration
  conn        Connection information
  ping        Connectivity testing
  dns         DNS lookup
  curl        curl examples
  nmap        Port scanning (requires nmap)
  tcpdump     Packet capture (requires root)
  bandwidth   Bandwidth tools
  all         Run all demos (default)

Note: Some commands require root privileges or additional tools.

EOF
}

# Main
case "${1:-all}" in
  config)    demo_config ;;
  conn)      demo_connections ;;
  ping)      demo_connectivity ;;
  dns)       demo_dns ;;
  curl)      demo_curl ;;
  nmap)      demo_nmap ;;
  tcpdump)   demo_tcpdump ;;
  bandwidth) demo_bandwidth ;;
  all)       run_all ;;
  -h|--help) usage ;;
  *) echo "Unknown section: $1"; usage; exit 1 ;;
esac
