====================
Bash Date cheatsheet
====================

.. contents:: Table of Contents
    :backlinks: none

Today
-----

.. code-block:: bash

    $ date
    Sun Jun 20 15:23:20 CST 2021
    $ date +"%Y%m%d"
    20210620

N Days Before
-------------

.. code-block:: bash

    # Linux
    $ date +%Y%m%d -d "1 day ago"
    20210619

    # BSD (MacOS)
    $ date -j -v-1d +"%Y%m%d"
    20210619
