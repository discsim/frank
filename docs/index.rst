.. Frankenstein documentation master file, created by
   sphinx-quickstart on Fri Jan 17 13:48:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: images/day_off.png
   :align: left
   :figwidth: 700

.. |br| raw:: html

   <br>

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
    Papers using frank <https://ui.adsabs.harvard.edu/search/q=citations(doi%3A10.1093%2Fmnras%2Fstaa1365)%20&sort=date%20desc%2C%20bibcode%20desc&p_=0>

.. toctree::
   :maxdepth: 1
   :caption: Under the hood

    API <py_API>
    Index <genindex>
    Github <https://github.com/discsim/frank>
    Submit an issue <https://github.com/discsim/frank/issues>

License & attribution
=====================

Frankenstein is free software licensed under the GPLv3 License.
For more details see the `LICENSE <https://github.com/discsim/frank/blob/master/LICENSE.txt>`_.

If you use frank for your research, please cite Jennings, Booth, Tazzari et al. 2020 MNRAS 495(3) 3209
`[MNRAS] <https://academic.oup.com/mnras/article/495/3/3209/5838058?guestAccessKey=7f163a1f-c12f-4771-8e54-928636794a5b>`_
`[ADS] <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract>`_
`[arXiv] <https://arxiv.org/abs/2005.07709>`_
`[Zenodo] <https://doi.org/10.5281/zenodo.3832064>`_:

.. code-block:: bash

    @ARTICLE{2020MNRAS.495.3209J,
    author = {{Jennings}, Jeff and {Booth}, Richard A. and {Tazzari}, Marco and {Rosotti}, Giovanni P. and {Clarke}, Cathie J.},
    title = "{frankenstein: protoplanetary disc brightness profile reconstruction at sub-beam resolution with a rapid Gaussian process}",
    journal = {\mnras},
    keywords = {methods: data analysis, protoplanetary discs, techniques: interferometric, planets and satellites: detection, submillimetre: general, submillimetre: planetary systems, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
    year = 2020,
    month = jul,
    volume = {495},
    number = {3},
    pages = {3209-3232},
    doi = {10.1093/mnras/staa1365},
    archivePrefix = {arXiv},
    eprint = {2005.07709},
    primaryClass = {astro-ph.EP},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

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
