=====================
Bash Basic cheatsheet
=====================

Check Command Exist
-------------------

.. code-block:: bash

    cmd="tput"
    if command -v "${tput}" > /dev/null; then
      echo "$cmd exist"
    else
      echo "$cmd does not exist"
    fi


Parse Arguments
---------------

.. code-block:: bash

	#!/bin/bash

	program="$1"

	usage() {
	  cat <<EOF

	Usage:	$program [OPTIONS] params

	Options:

	  -h,--help                show this help
	  -a,--argument string     set an argument

	EOF
	}

	arg=""
	params=""
	while (( "$#" )); do
	  case "$1" in
		-h|-\?|--help)
		  usage
		  exit 0
		  ;;

		-a|--argument)
		  arg="$2"
		  shift 2
		  ;;

		# stop parsing
		--)
		  shift
		  break
		  ;;

		# unsupport options
		-*|--*=)
		  echo "Error: unsupported option $1" >&2
		  exit 1
		  ;;

		# positional arguments
		*)
		  params="$params $1"
		  shift
		  ;;
	  esac
	done
