=======
Systemd
=======

.. contents:: Table of Contents
    :backlinks: none

Services Management
-------------------

.. code-block:: bash

   # start
   $ systemctl start app.service

   # stop
   $ systemctl stop app.service

   # restart
   $ systemctl restart app.service

   # reload
   $ systemctl reload app.service

   # reload a daemon after modifying a unit
   $ systemctl daemon-reload

   # enable a service
   $ systemctl enable app.service

   # disable a service
   $ systemctl disable app.service

   # check status
   $ systemctl status app.service

   # check is active
   $ systemctl is-active app.service

   # check is enabled
   $ systemctl is-enabled app.service

   # list all units
   $ systemctl list-units

   # list all timers
   $ systemctl list-timers

User Services
-------------

.. code-block:: bash

   $ sudo loginctl enable-linger $USER
   $ mkdir -p ~/.config/systemd/user/

   # allow journalctl --user -xe -f --all
   $ vim /etc/systemd/journald.conf

   [Journal]
   Storage=persistent

   # move app.service to ~/.config/systemd/user/
   $ systemctl --user start app.service

Service Unit
------------

.. code-block:: bash

    # app.service
    #
    # $ systemctl enable app.service
    # $ systemctl start app.service
    # $ systemctl status app.service

    [Unit]
    Description=Run an application

    [Service]
    Type=simple
    Restart=always
    RestartSec=30
    WorkingDirectory=/path/to/app
    ExecStart=/bin/bash run.sh

    [Install]
    WantedBy=multi-user.target

Timer Unit
----------

.. code-block:: bash

    # job.timer
    #
    # $ systemctl enable job.timer
    # $ systemctl start job.timer
    # $ systemctl list-timers

    [Unit]
    Description=Run a timer

    [Timer]
    OnBootSec=10min
    OnUnitActiveSec=1m
    Unit=job.service

    [Install]
    WantedBy=multi-user.target

.. code-block:: bash

    # job.service

    [Unit]
    Description=Run a job

    [Service]
    Type=oneshot
    WorkingDirectory=/path/to/job/folder
    ExecStart=/bin/bash run.sh

    [Install]
    WantedBy=multi-user.target



.. code-block::

              Minimal form              Normalized form
      Sat,Thu,Mon..Wed,Sat..Sun → Mon..Thu,Sat,Sun *-*-* 00:00:00
          Mon,Sun 12-*-* 2,1:23 → Mon,Sun 2012-*-* 01,02:23:00
                        Wed *-1 → Wed *-*-01 00:00:00
               Wed..Wed,Wed *-1 → Wed *-*-01 00:00:00
                     Wed, 17:48 → Wed *-*-* 17:48:00
    Wed..Sat,Tue 12-10-15 1:2:3 → Tue..Sat 2012-10-15 01:02:03
                    *-*-7 0:0:0 → *-*-07 00:00:00
                          10-15 → *-10-15 00:00:00
            monday *-12-* 17:00 → Mon *-12-* 17:00:00
      Mon,Fri *-*-3,1,2 *:30:45 → Mon,Fri *-*-01,02,03 *:30:45
           12,14,13,12:20,10,30 → *-*-* 12,13,14:10,20,30:00
                12..14:10,20,30 → *-*-* 12..14:10,20,30:00
      mon,fri *-1/2-1,3 *:30:45 → Mon,Fri *-01/2-01,03 *:30:45
                 03-05 08:05:40 → *-03-05 08:05:40
                       08:05:40 → *-*-* 08:05:40
                          05:40 → *-*-* 05:40:00
         Sat,Sun 12-05 08:05:40 → Sat,Sun *-12-05 08:05:40
               Sat,Sun 08:05:40 → Sat,Sun *-*-* 08:05:40
               2003-03-05 05:40 → 2003-03-05 05:40:00
     05:40:23.4200004/3.1700005 → *-*-* 05:40:23.420000/3.170001
                 2003-02..04-05 → 2003-02..04-05 00:00:00
           2003-03-05 05:40 UTC → 2003-03-05 05:40:00 UTC
                     2003-03-05 → 2003-03-05 00:00:00
                          03-05 → *-03-05 00:00:00
                         hourly → *-*-* *:00:00
                          daily → *-*-* 00:00:00
                      daily UTC → *-*-* 00:00:00 UTC
                        monthly → *-*-01 00:00:00
                         weekly → Mon *-*-* 00:00:00
        weekly Pacific/Auckland → Mon *-*-* 00:00:00 Pacific/Auckland
                         yearly → *-01-01 00:00:00
                       annually → *-01-01 00:00:00
                          *:2/3 → *-*-* *:02/3:00
