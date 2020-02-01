Installation
============

With pip
--------

Install the latest stable version of Frankenstein (``frank``) with `pip <https://pip.pypa.io/en/stable/>`_,

.. code-block:: bash

    pip install frank

From source
-----------

Clone the source repository from `GitHub <https://github.com/discsim/frank>`_ if you're feeling a bit monstrous,

.. code-block:: bash

    pip install git+https://github.com/discsim/frank.git

Test the install
################

If you cloned the source repo and you have `py.test <https://docs.pytest.org/en/latest/>`_,
just run this from the code's root directory (it takes ~5 sec),

.. code-block:: bash

    py.test frank/tests.py
