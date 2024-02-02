====================
Bash Find cheatsheet
====================

.. contents:: Table of Contents
    :backlinks: none


Find by Suffix
--------------

.. code-block:: bash

    $ find "${path}" -name "*.py"

Find by Substring
-----------------

.. code-block:: bash

    $ find "${path}" -name "*code*"

Find by Case Insensitive
------------------------

.. code-block:: bash

    $ find "${path}" -iname "*.py"

Find by File Type
-----------------

.. code-block:: bash

    # b  block
    # c  character
    # d  directory
    # p  named pipe
    # f  regular file
    # l  symbolic link
    # s  socket

    # find regular file
    $ find "${path}" -type f -name "*.py"

    # find directory
    $ find "${path}" -type d

Find by Size
------------

.. code-block:: bash

    # find files < 50M
    $ find "${path}" -type f -size -50M

    # find files > 50M
    $ find "${path}" -type f -size +50M

Find by Date
------------

.. code-block:: bash

    # files are not accessed > 7 days
    $ find "${path}" -type f -atime +7

    # files are accessed < 7 days
    $ find "${path}" -type f -atime -7

    # files are not accessed > 10 min
    $ find "${path}" -type f -amin +10

    # files are accessed < 10 min
    $ find "${path}" -type f -amin -10

Find by User
------------

.. code-block:: bash

    $ find "${path}" -type f -user "${USER}"

Delete after Find
-----------------

.. code-block:: bash

    # delete by pattern
    $ find "${path}" -type f -name "*.sh" -delete

    # delete recursively
    find ker -type d -exec rm -rf {} \+

Loop through files
------------------

.. code-block:: bash

   # ref: https://stackoverflow.com/questions/9612090

   # execute `echo` once for each file
   find "${path}" -name "*.txt" -exec echo {} \;

   # execute `echo` once with all the files
   find "${path}" -name "*.txt" -exec echo {} +

   # using while loop
   find "${path}" -name "*.txt" -print0 | while IFS= read -r -d '' file; do
     echo "$file"
   done

``grep`` after find
-------------------

.. code-block:: bash

    $ find ker -type f -exec grep -rni "test" {} \+

    # or

    $ find ker -type f -exec grep -rni "test" {} \;
