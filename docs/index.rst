.. Frankenstein documentation master file, created by
   sphinx-quickstart on Fri Jan 17 13:48:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: images/day_off.png
   :align: left
   :figwidth: 700

Documentation
=============

Welcome to the Frankenstein (``frank``) documentation, you monster. frank fits the 1D radial brightness profile of an
interferometric source given a set of visibilities. This site details how to
install and run the code, provides examples for applying the model and
interpreting its results, and hosts the code's API.

.. toctree::
   :maxdepth: 1
   :caption: Using the code

    Installation <install>
    Quickstart <quickstart>
    Tutorials <tutorials>
    Papers using frank <https://ui.adsabs.harvard.edu/public-libraries/xx>

.. toctree::
   :maxdepth: 1
   :caption: Under the hood

    API <py_API>
    Index <genindex>
    Github <https://github.com/discsim/frank>
    Submit an issue <https://github.com/discsim/frank/issues>

Notes
=====
``frank`` requires Python >= 3.0 and does not support compatibility with Python 2.x. It also uses ``numpy``, ``scipy`` and ``matplotlib`` (see ``requirements.txt``).

License & attribution
=====================

Frankenstein is free software licensed under the GPLv3 License.
For more details see the `LICENSE <https://github.com/discsim/frank/blob/master/LICENSE.txt>`_.

If you use frank for your research, please cite Jennings, Booth, Tazzari et al. (2020) MNRAS **xx** xx [`MNRAS <xx>`_] [`arXiv <xx>`_] [`ADS <xx>`_].
The `Zenodo reference <https://zenodo.org/badge/latestdoi/xxx>`_ is  ::

    @misc{}xx update ADS bibtex entry xx

Authors
-------

    - `Richard 'Dr. Frankenstein' Booth (University of Cambridge) <https://github.com/rbooth200>`_
    - `Jeff 'The Monster' Jennings (University of Cambridge) <https://github.com/jeffjennings>`_
    - `Marco 'It's Alive!!!' Tazzari (University of Cambridge) <https://github.com/mtazzari>`_

Contact
#######
Interested in collaborating to improve, extend or apply frank?
Or just have questions about the code that don't require submitting an issue on GitHub?
`Email Jeff! <jmj51@ast.cam.ac.uk>`_
