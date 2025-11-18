==================================
Bash Regular Expression Cheatsheet
==================================

.. contents:: Table of Contents
    :backlinks: none

``grep`` vs ``grep -E``
-----------------------

The difference between grep and grep -E is that grep uses basic regular
expressions while grep -E uses extended regular expressions. In basic
regular expressions, the characters "?", "+", "{", "|", "(",")" lose their
special meaning; instead, use "?", "+", "{", "|", "(", ")".


.. code-block:: bash

    $ echo "987-123-4567" | grep "^[0-9]\{3\}-[0-9]\{3\}-[0-9]\{4\}$"
    $ echo "987-123-4567" | grep -E "^[0-9]{3}-[0-9]{3}-[0-9]{4}$"



`tr` Substitutes Strings
------------------------


.. code-block:: bash

    # tr substitutes white spaces to newline
    $ echo "a b c" | tr "[:space:]+" "\n"
    a
    b
    c

    # tr spueeze multiple spaces
    $ echo "a    b   c" | tr -s " "
    a b c

`uniq` Filters out Repeated Lines
---------------------------------

.. code-block:: bash

    $ echo "a a b b c" | tr " " "\n" | sort | uniq
    a
    b
    c

    # display count
    $ echo "a a b b a c" | tr " " "\n" | sort | uniq -c
       3 a
       2 b
       1 c

Note that ``uniq`` only filters out lines continuously. However, if characters
are equal but they does not appear continually, ``uniq`` does not squeeze them.
Therefore, a programmer needs to use ``sort`` to categorizes lines before
``uniq``.

.. code-block:: bash

    $ echo "a a b b a c" | tr " " "\n" | uniq
    a
    b
    a
    c

``sort`` lines
--------------

.. code-block:: bash

    # sort by lines
    $ echo "b a c d" | tr " " "\n" | sort

    # sort by lines reversely
    $ echo "b a c d" | tr " " "\n" | sort -r

    # sort by a field
    $ echo "b a b c d" | tr " " "\n" | sort | uniq -c | sort -k1
