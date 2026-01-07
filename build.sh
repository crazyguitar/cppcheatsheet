#!/bin/bash
#
# Build script for Libefaxx project
#
# This script configures and builds the project using CMake and Ninja.
# It supports clean builds and parallel compilation.
#

set -uo pipefail

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROGRAM="$0"
CLEAN=false
JOBS=$(nproc)

# Log info message with timestamp
info() {
  echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][info] $*"
}

# Log error message with timestamp to stderr
err() {
  echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2
}

# Display usage information
usage() {
  cat <<EOF
Usage:  $PROGRAM [OPTIONS]
Options:
  -h,--help                  show this help
  -c,--clean                 clean build
  -j [N],--jobs [N]          specify the number of jobs

EOF
}

# Build the project using CMake and Ninja
# Args:
#   $1 - Number of parallel jobs
#   $2 - Whether to clean build directory first (true/false)
build() {
  local jobs="$1"
  local clean="$2"
  local dir="${DIR}/build"

  if [ "${clean}" == true ]; then
    rm -rf "${dir}"
  fi

  set -x

  if ! cmake -GNinja -B "${dir}"; then
    err "run cmake configuration failed."
    return 1
  fi

  if ! cmake --build "${dir}" -j "${jobs}" --verbose; then
    err "build source code faild."
    return 1
  fi
}

while (( "$#" )); do
  case "$1" in
    -h|-\?|--help) usage; exit 0 ;;
    -j|--jobs) JOBS="${2}"; shift 2 ;;
    -c|--clean) CLEAN=true; shift ;;
    --*=|-*) err "unsupported option $1"; exit 1 ;;
  esac
done

if ! build "${JOBS}" "${CLEAN}"; then
  exit 1
fi
