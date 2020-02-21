Installation
============

With pip
--------

Install the latest stable version of Frankenstein (``frank``) with `pip <https://pypi.org/project/frank/>`_,

.. code-block:: bash

    pip install frank

Then play `this <https://drive.google.com/file/d/1SEz8YqB2rRS1uMguXxI1RI7Jk27yQfLO/view?usp=sharing>`_ loud.

From source
-----------

.. role:: strike

Join us in our efforts to :strike:`revive the dead` do very moral science.
Clone the source repository from `GitHub <https://github.com/discsim/frank>`_,

.. code-block:: bash

    pip install git+https://github.com/discsim/frank.git

(this is the same as `git clone https://github.com/discsim/frank.git; cd frank`).

Test the install
################

If you cloned the source repo and you have `py.test <https://docs.pytest.org/en/latest/>`_,
run it from the code's root directory (it takes <1 min),

.. code-block:: bash

    py.test frank/tests.py
