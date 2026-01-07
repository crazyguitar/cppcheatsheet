=======
Systemd
=======

.. meta::
   :description: Complete systemd tutorial and cheatsheet for Linux service management, unit files, timers, journalctl logging, coredumpctl debugging, and boot optimization.
   :keywords: systemd, systemctl, Linux service, unit file, timer, journalctl, coredumpctl, core dump, cron replacement, init system, daemon, boot, service management

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Systemd is the standard init system and service manager for most modern Linux
distributions including Ubuntu, Debian, Fedora, CentOS, and Arch Linux. It
replaces the traditional SysV init system with a more powerful and flexible
approach to managing system startup, services, logging, and system state.

Systemd uses unit files to define services, timers, mount points, and other
system resources. These declarative configuration files specify dependencies,
execution parameters, and restart behavior, making service management more
predictable and easier to debug than shell scripts.

Services Management
-------------------

The ``systemctl`` command is the primary tool for controlling systemd services.
It allows administrators to start, stop, restart, enable, and disable services,
as well as query their status and manage dependencies. Understanding these
commands is essential for Linux system administration.

.. code-block:: bash

    # Start/stop/restart a service
    $ systemctl start app.service
    $ systemctl stop app.service
    $ systemctl restart app.service

    # Reload service configuration (without restart)
    $ systemctl reload app.service

    # Reload systemd after modifying unit files
    $ systemctl daemon-reload

    # Enable/disable service at boot
    $ systemctl enable app.service
    $ systemctl disable app.service

    # Check service status
    $ systemctl status app.service
    $ systemctl is-active app.service
    $ systemctl is-enabled app.service

    # List units
    $ systemctl list-units                    # all loaded units
    $ systemctl list-units --type=service     # only services
    $ systemctl list-units --state=failed     # failed units
    $ systemctl list-unit-files               # all installed unit files

    # Show unit dependencies
    $ systemctl list-dependencies app.service

    # Mask/unmask (completely disable a service)
    $ systemctl mask app.service
    $ systemctl unmask app.service

Viewing Logs with journalctl
----------------------------

Systemd includes a centralized logging system called the journal, which collects
logs from all services, the kernel, and system messages. The ``journalctl``
command provides powerful filtering and querying capabilities that make
troubleshooting much easier than parsing traditional log files in ``/var/log``.

The journal stores logs in a binary format with rich metadata including
timestamps, priority levels, unit names, and process IDs. This enables efficient
filtering by service, time range, or severity level.

.. code-block:: bash

    # View logs for a service
    $ journalctl -u app.service

    # Follow logs in real-time (like tail -f)
    $ journalctl -u app.service -f

    # Show logs since last boot
    $ journalctl -u app.service -b

    # Show logs from last hour
    $ journalctl -u app.service --since "1 hour ago"

    # Show logs between dates
    $ journalctl -u app.service --since "2024-01-01" --until "2024-01-02"

    # Filter by priority (emerg, alert, crit, err, warning, notice, info, debug)
    $ journalctl -u app.service -p err

    # Show kernel messages (like dmesg)
    $ journalctl -k

    # Show disk usage by journal
    $ journalctl --disk-usage

    # Clean old logs (keep last 1G)
    $ journalctl --vacuum-size=1G

    # Clean logs older than 2 weeks
    $ journalctl --vacuum-time=2weeks

Service Unit File
-----------------

A systemd service unit file is a configuration file that defines how to manage
a background service or daemon. Unit files use an INI-style format with three
main sections: ``[Unit]`` for metadata and dependencies, ``[Service]`` for
process execution parameters, and ``[Install]`` for installation targets.

Place custom unit files in ``/etc/systemd/system/`` for system services. After
creating or modifying a unit file, run ``systemctl daemon-reload`` to reload
the systemd configuration.

.. code-block:: ini

    # /etc/systemd/system/app.service
    #
    # $ systemctl daemon-reload
    # $ systemctl enable app.service
    # $ systemctl start app.service

    [Unit]
    Description=My Application
    After=network.target
    Wants=network-online.target

    [Service]
    Type=simple
    User=appuser
    Group=appgroup
    WorkingDirectory=/path/to/app
    ExecStart=/usr/bin/python3 app.py
    ExecReload=/bin/kill -HUP $MAINPID
    Restart=on-failure
    RestartSec=10

    # Environment variables
    Environment=NODE_ENV=production
    EnvironmentFile=/etc/app/env

    # Security hardening options
    NoNewPrivileges=true
    ProtectSystem=strict
    ProtectHome=true
    PrivateTmp=true

    [Install]
    WantedBy=multi-user.target

Service Types
~~~~~~~~~~~~~

The ``Type=`` directive tells systemd how the service notifies that it has
finished starting. Choosing the correct type ensures systemd accurately tracks
the service state and manages dependencies properly.

.. code-block:: text

    Type=simple      Default. Process started by ExecStart is the main process.
    Type=forking     Process forks and parent exits. Use with PIDFile=.
    Type=oneshot     Process exits after completing. Use for scripts.
    Type=notify      Like simple, but service sends notification when ready.
    Type=idle        Like simple, but waits until other jobs finish.

Restart Policies
~~~~~~~~~~~~~~~~

The ``Restart=`` directive controls when systemd automatically restarts a
service after it exits. Use ``RestartSec=`` to add a delay between restart
attempts, preventing rapid restart loops that could overwhelm the system.

.. code-block:: text

    Restart=no           Don't restart (default)
    Restart=on-success   Restart only on clean exit (exit code 0)
    Restart=on-failure   Restart on non-zero exit, signal, timeout
    Restart=on-abnormal  Restart on signal, timeout, watchdog
    Restart=always       Always restart regardless of exit status

Timer Unit File
---------------

Systemd timers are a modern replacement for cron jobs, offering better
integration with the service manager, logging, and dependency handling. A timer
unit activates a corresponding service unit based on time events. Unlike cron,
timers can trigger on boot, on service activation, or on calendar schedules.

Each timer requires a matching service file with the same base name (e.g.,
``backup.timer`` triggers ``backup.service``).

Managing Timers
~~~~~~~~~~~~~~~

.. code-block:: bash

    # List all active timers with next/last run times
    $ systemctl list-timers

    # List all timers including inactive
    $ systemctl list-timers --all

    # Show when a specific timer will next execute
    $ systemctl list-timers backup.timer

    # Example output:
    # NEXT                        LEFT          LAST                        PASSED       UNIT           ACTIVATES
    # Mon 2024-01-08 02:00:00 UTC 5h left       Sun 2024-01-07 02:00:00 UTC 18h ago      backup.timer   backup.service

    # Enable and start a timer
    $ systemctl enable backup.timer
    $ systemctl start backup.timer

    # Check timer status
    $ systemctl status backup.timer

    # Manually trigger the associated service (for testing)
    $ systemctl start backup.service

    # View timer logs
    $ journalctl -u backup.timer
    $ journalctl -u backup.service

Timer Unit Example
~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    # /etc/systemd/system/backup.timer
    #
    # $ systemctl daemon-reload
    # $ systemctl enable backup.timer
    # $ systemctl start backup.timer

    [Unit]
    Description=Daily backup timer

    [Timer]
    OnCalendar=*-*-* 02:00:00
    Persistent=true
    RandomizedDelaySec=1h

    [Install]
    WantedBy=timers.target

.. code-block:: ini

    # /etc/systemd/system/backup.service

    [Unit]
    Description=Backup job

    [Service]
    Type=oneshot
    ExecStart=/usr/local/bin/backup.sh

Timer Options
~~~~~~~~~~~~~

Timers support two types of triggers: monotonic timers that fire relative to
a specific event (boot, activation), and realtime timers that fire based on
calendar time. Use ``Persistent=true`` to run missed jobs after system downtime.

.. code-block:: text

    OnBootSec=10min       Run 10 minutes after boot
    OnUnitActiveSec=1h    Run 1 hour after service last activated
    OnCalendar=daily      Run daily at midnight
    OnCalendar=weekly     Run weekly on Monday at midnight
    OnCalendar=hourly     Run every hour
    OnCalendar=*:0/15     Run every 15 minutes
    Persistent=true       Run immediately if missed while system was off
    RandomizedDelaySec=   Add random delay to prevent thundering herd

Calendar Syntax Examples
~~~~~~~~~~~~~~~~~~~~~~~~

Systemd calendar expressions use the format ``DayOfWeek Year-Month-Day
Hour:Minute:Second``. Use ``*`` as a wildcard for any value and ``/`` for
intervals. The ``systemd-analyze calendar`` command validates and normalizes
calendar expressions.

.. code-block:: text

    *-*-* 00:00:00        Daily at midnight
    *-*-* *:00:00         Every hour
    *-*-* *:*:00          Every minute
    Mon *-*-* 00:00:00    Every Monday at midnight
    Mon,Fri *-*-* 17:00   Monday and Friday at 5pm
    *-*-01 00:00:00       First day of every month
    *-01-01 00:00:00      January 1st every year
    2024-*-* 00:00:00     Every day in 2024

    # Test and validate calendar expressions
    $ systemd-analyze calendar "Mon *-*-* 09:00"

User Services
-------------

Systemd supports user-level services that run without root privileges in the
user's session. User services are ideal for personal daemons, development
servers, or user-specific background tasks like syncing or notifications.

By default, user services only run while the user is logged in. Enable
lingering to keep services running after logout, which is essential for
servers or long-running background tasks.

.. code-block:: bash

    # Enable lingering (services run without login)
    $ sudo loginctl enable-linger $USER

    # Create user service directory
    $ mkdir -p ~/.config/systemd/user/

    # Place unit files in ~/.config/systemd/user/
    # Manage with --user flag
    $ systemctl --user daemon-reload
    $ systemctl --user enable app.service
    $ systemctl --user start app.service
    $ systemctl --user status app.service

    # View user service logs
    $ journalctl --user -u app.service -f

Enable persistent journal storage to preserve user service logs across reboots:

.. code-block:: bash

    $ sudo vim /etc/systemd/journald.conf

    [Journal]
    Storage=persistent

    $ sudo systemctl restart systemd-journald

Analyzing Boot Performance
--------------------------

Systemd provides built-in tools to analyze boot performance and identify slow
services that delay system startup. Use these commands to optimize boot time
and diagnose startup issues.

.. code-block:: bash

    # Show total boot time breakdown
    $ systemd-analyze

    # Show time taken by each unit (sorted by duration)
    $ systemd-analyze blame

    # Show critical chain (services blocking boot)
    $ systemd-analyze critical-chain

    # Generate visual boot chart as SVG
    $ systemd-analyze plot > boot.svg

Core Dumps with coredumpctl
---------------------------

Systemd includes ``systemd-coredump`` which automatically captures and stores
core dumps when processes crash. The ``coredumpctl`` command lets you list,
inspect, and debug these core dumps. This is invaluable for debugging crashes
in C/C++ applications running as systemd services.

.. code-block:: bash

    # List all core dumps
    $ coredumpctl list

    # List core dumps for a specific executable
    $ coredumpctl list /usr/bin/myapp

    # Show info about the most recent core dump
    $ coredumpctl info

    # Show info for a specific PID
    $ coredumpctl info 12345

    # Launch debugger (gdb) on most recent core dump
    $ coredumpctl debug

    # Debug a specific executable's core dump
    $ coredumpctl debug /usr/bin/myapp

    # Export core dump to a file
    $ coredumpctl dump -o core.dump

    # Export core dump for specific PID
    $ coredumpctl dump 12345 -o core.dump

Configure core dump storage in ``/etc/systemd/coredump.conf``:

.. code-block:: ini

    [Coredump]
    Storage=external          # external, journal, or none
    Compress=yes              # compress stored core dumps
    MaxUse=1G                 # max disk space for core dumps
    KeepFree=1G               # min free space to maintain
    ProcessSizeMax=2G         # max size of process to dump

Useful Paths
------------

.. code-block:: text

    /etc/systemd/system/          System unit files (admin-created)
    /usr/lib/systemd/system/      System unit files (package-installed)
    /run/systemd/system/          Runtime unit files (temporary)
    ~/.config/systemd/user/       User unit files
    /etc/systemd/journald.conf    Journal configuration
    /etc/systemd/coredump.conf    Core dump configuration
    /etc/systemd/system.conf      System manager configuration
    /var/lib/systemd/coredump/    Stored core dumps
