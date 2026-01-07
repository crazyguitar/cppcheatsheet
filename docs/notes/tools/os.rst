===========================
Operating System Cheatsheet
===========================

.. meta::
   :description: Linux/Unix command-line reference covering date manipulation, file searching with find, system information, process management, and file operations.
   :keywords: Linux commands, Unix utilities, find command, date command, system administration, process management, bash utilities

.. contents:: Table of Contents
    :backlinks: none

From date manipulation to process control, mastering command-line utilities is essential
for effective system administration. Here you'll find practical examples for everyday
tasks: parsing dates, searching files with find, managing processes, and handling
compression—all organized for quick reference.

An interactive demo script is available at `src/bash/os.sh <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/bash/os.sh>`_
to help you experiment with the concepts covered in this cheatsheet.

.. code-block:: bash

    ./src/bash/os.sh           # Run all demos
    ./src/bash/os.sh date      # Run date/time demo
    ./src/bash/os.sh find      # Run find command demo
    ./src/bash/os.sh --help    # Show all available sections

Date and Time
=============

The ``date`` command displays and manipulates system date and time. Format strings
control output, while GNU and BSD versions differ in syntax for date arithmetic.

Current Date and Time
---------------------

.. code-block:: bash

    date                        # Full date and time
    date +"%Y-%m-%d"            # 2024-01-15
    date +"%Y%m%d"              # 20240115
    date +"%H:%M:%S"            # 14:30:45
    date +"%Y-%m-%d %H:%M:%S"   # 2024-01-15 14:30:45
    date -Iseconds              # ISO 8601 format with timezone

Common Format Specifiers
------------------------

.. code-block:: bash

    %Y    # Year (2024)
    %m    # Month (01-12)
    %d    # Day (01-31)
    %H    # Hour 24h (00-23)
    %M    # Minute (00-59)
    %S    # Second (00-59)
    %s    # Unix timestamp
    %A    # Weekday name (Monday)
    %B    # Month name (January)
    %Z    # Timezone (PST)

Date Arithmetic
---------------

.. code-block:: bash

    # Linux (GNU date)
    date +%Y-%m-%d -d "1 day ago"       # Yesterday
    date +%Y-%m-%d -d "7 days ago"      # 7 days ago
    date +%Y-%m-%d -d "1 month ago"     # 1 month ago
    date +%Y-%m-%d -d "next monday"     # Next Monday
    date +%Y-%m-%d -d "2024-01-15 + 10 days"

    # BSD/macOS
    date -j -v-1d +"%Y-%m-%d"           # Yesterday
    date -j -v-7d +"%Y-%m-%d"           # 7 days ago
    date -j -v-1m +"%Y-%m-%d"           # 1 month ago
    date -j -v+10d +"%Y-%m-%d"          # 10 days from now

Convert Timestamps
------------------

.. code-block:: bash

    # Unix timestamp to date (Linux)
    date -d @1705334400

    # Unix timestamp to date (BSD/macOS)
    date -r 1705334400

    # Current timestamp
    date +%s

File Searching with find
========================

The ``find`` command recursively searches directories for files matching specified
criteria. It supports filtering by name, type, size, time, ownership, and permissions,
with actions to execute on matched files.

Basic Patterns
--------------

.. code-block:: bash

    find /path -name "*.py"             # By extension
    find /path -name "*test*"           # By substring
    find /path -iname "*.PY"            # Case insensitive
    find /path -path "**/test/*"        # By path pattern

File Types
----------

.. code-block:: bash

    find /path -type f                  # Regular files
    find /path -type d                  # Directories
    find /path -type l                  # Symbolic links
    find /path -type f -name "*.py"     # Combine filters

    # Type codes: f=file, d=directory, l=symlink,
    #             b=block, c=character, p=pipe, s=socket

By Size
-------

.. code-block:: bash

    find /path -type f -size +100M      # Larger than 100MB
    find /path -type f -size -1M        # Smaller than 1MB
    find /path -type f -size 50M        # Exactly 50MB

    # Size units: c=bytes, k=KB, M=MB, G=GB

By Time
-------

.. code-block:: bash

    # Access time
    find /path -type f -atime +7        # Accessed > 7 days ago
    find /path -type f -atime -7        # Accessed < 7 days ago
    find /path -type f -amin +60        # Accessed > 60 min ago

    # Modification time
    find /path -type f -mtime +30       # Modified > 30 days ago
    find /path -type f -mmin -10        # Modified < 10 min ago

    # Change time (metadata)
    find /path -type f -ctime +7        # Changed > 7 days ago

    # Newer than file
    find /path -type f -newer ref.txt

By Ownership and Permissions
----------------------------

.. code-block:: bash

    find /path -type f -user root       # Owned by root
    find /path -type f -group admin     # Group admin
    find /path -type f -perm 644        # Exact permissions
    find /path -type f -perm -644       # At least these permissions
    find /path -type f -perm /u+x       # User executable

Combining Conditions
--------------------

.. code-block:: bash

    # AND (implicit)
    find /path -type f -name "*.log" -size +1M

    # OR
    find /path \( -name "*.py" -o -name "*.sh" \)

    # NOT
    find /path -type f ! -name "*.txt"

    # Complex
    find /path -type f \( -name "*.py" -o -name "*.sh" \) ! -path "*/venv/*"

Actions
-------

.. code-block:: bash

    # Delete matched files
    find /path -type f -name "*.tmp" -delete

    # Execute command for each file
    find /path -type f -name "*.sh" -exec chmod +x {} \;

    # Execute command with all files
    find /path -type f -name "*.txt" -exec grep -l "pattern" {} +

    # Delete directories recursively
    find /path -type d -name "__pycache__" -exec rm -rf {} +

    # Print with null delimiter (for xargs)
    find /path -type f -print0 | xargs -0 ls -la

Loop Through Results
--------------------

.. code-block:: bash

    # Using while loop (handles spaces in filenames)
    find /path -type f -name "*.txt" -print0 | while IFS= read -r -d '' file; do
      echo "Processing: $file"
    done

    # Process substitution (preserves variables)
    count=0
    while IFS= read -r -d '' file; do
      ((count++))
    done < <(find /path -type f -print0)
    echo "Found: $count files"

    # Sort results
    find /path -name "*.txt" -print0 | sort -z | xargs -r0 -I{} echo "{}"

System Information
==================

Commands for gathering system software and configuration information.

System Information
------------------

.. code-block:: bash

    uname -a                            # All system info
    uname -s                            # Kernel name
    uname -r                            # Kernel release
    uname -m                            # Machine architecture
    hostname                            # System hostname
    uptime                              # System uptime
    cat /etc/os-release                 # OS info (Linux)
    sw_vers                             # macOS version

Process Management
==================

Commands for monitoring and controlling system processes.

Viewing Processes
-----------------

.. code-block:: bash

    ps aux                              # All processes
    ps aux | grep nginx                 # Filter by name
    ps -ef                              # Full format listing
    ps -p PID -o pid,ppid,cmd,%cpu,%mem # Specific process details
    pgrep -l nginx                      # Find PID by name
    pstree                              # Process tree
    top                                 # Interactive process viewer
    htop                                # Enhanced top (if installed)

Process Control
---------------

.. code-block:: bash

    kill PID                            # Terminate (SIGTERM)
    kill -9 PID                         # Force kill (SIGKILL)
    kill -HUP PID                       # Reload config (SIGHUP)
    killall nginx                       # Kill by name
    pkill -f "python script.py"         # Kill by pattern

Background Jobs
---------------

.. code-block:: bash

    command &                           # Run in background
    jobs                                # List background jobs
    fg %1                               # Bring job 1 to foreground
    bg %1                               # Resume job 1 in background
    disown %1                           # Detach job from shell
    nohup command &                     # Run immune to hangups

File Operations
===============

Commands for file manipulation, compression, and text processing.

File Management
---------------

.. code-block:: bash

    cp -r src dest                      # Copy recursively
    mv old new                          # Move/rename
    rm -rf dir                          # Remove recursively
    mkdir -p a/b/c                      # Create nested dirs
    ln -s target link                   # Symbolic link
    chmod 755 file                      # Change permissions
    chown user:group file               # Change ownership

Compression
-----------

.. code-block:: bash

    # tar
    tar -cvf archive.tar dir/           # Create tar
    tar -xvf archive.tar                # Extract tar
    tar -czvf archive.tar.gz dir/       # Create gzipped tar
    tar -xzvf archive.tar.gz            # Extract gzipped tar
    tar -cjvf archive.tar.bz2 dir/      # Create bz2 tar
    tar -tzvf archive.tar.gz            # List contents

    # zip
    zip -r archive.zip dir/             # Create zip
    unzip archive.zip                   # Extract zip
    unzip -l archive.zip                # List contents

    # gzip
    gzip file                           # Compress (replaces file)
    gunzip file.gz                      # Decompress
    zcat file.gz                        # View without extracting

Viewing Files
-------------

.. code-block:: bash

    cat file                            # Print entire file
    head -n 20 file                     # First 20 lines
    tail -n 20 file                     # Last 20 lines
    tail -f file                        # Follow file updates
    less file                           # Paginated view
    wc -l file                          # Line count
    wc -w file                          # Word count

File Comparison
---------------

.. code-block:: bash

    diff file1 file2                    # Line differences
    diff -u file1 file2                 # Unified format
    diff -r dir1 dir2                   # Compare directories
    cmp file1 file2                     # Byte comparison
    comm file1 file2                    # Common/unique lines (sorted)
    md5sum file                         # MD5 checksum
    sha256sum file                      # SHA256 checksum

Environment and Shell
=====================

Commands for managing shell environment and configuration.

Environment Variables
---------------------

.. code-block:: bash

    env                                 # All environment variables
    printenv PATH                       # Specific variable
    export VAR=value                    # Set variable
    unset VAR                           # Remove variable
    echo $PATH                          # Print variable

Command Information
-------------------

.. code-block:: bash

    which command                       # Command location
    type command                        # Command type
    whereis command                     # Binary/man/source locations
    man command                         # Manual page
    command --help                      # Help text
    alias                               # List aliases
    alias ll='ls -la'                   # Create alias

History
-------

.. code-block:: bash

    history                             # Command history
    history | grep pattern              # Search history
    !n                                  # Execute command n
    !!                                  # Repeat last command
    !$                                  # Last argument of previous command
    ctrl+r                              # Reverse search
