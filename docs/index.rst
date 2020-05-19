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

If you use frank for your research, please cite Jennings, Booth, Tazzari et al. 2020 MNRAS (accepted)
`[MNRAS] <https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staa1365/5838058?guestAccessKey=7f163a1f-c12f-4771-8e54-928636794a5b>`_
`[arXiv] <https://arxiv.org/abs/2005.07709>`_
`[Zenodo] <https://doi.org/10.5281/zenodo.3832064>`_:

.. code-block:: bash

    @article{10.1093/mnras/staa1365,
    author = {Jennings, Jeff and Booth, Richard A and Tazzari, Marco and Rosotti, Giovanni P and Clarke, Cathie J},
    title = "{Frankenstein: Protoplanetary disc brightness profile reconstruction at sub-beam resolution with a rapid Gaussian process}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2020},
    month = {05},
    abstract = "{Interferometric observations of the mm dust distribution in protoplanetary discs are now showing a ubiquity of annular gap and ring substructures. Their identification and accurate characterization is critical to probing the physical processes responsible. We present Frankenstein (frank), an open source code that recovers axisymmetric disc structures at sub-beam resolution. By fitting the visibilities directly, the model reconstructs a disc’s 1D radial brightness profile nonparametrically using a fast (≲1 min) Gaussian process. The code avoids limitations of current methods that obtain the radial brightness profile by either extracting it from the disc image via nonlinear deconvolution at the cost of reduced fit resolution, or by assumptions placed on the functional forms of disc structures to fit the visibilities parametrically. We use mock ALMA observations to quantify the method’s intrinsic capability and its performance as a function of baseline-dependent signal-to-noise. Comparing the technique to profile extraction from a CLEAN image, we motivate how our fits accurately recover disc structures at a sub-beam resolution. Demonstrating the model’s utility in fitting real high and moderate resolution observations, we conclude by proposing applications to address open questions on protoplanetary disc structure and processes.}",
    issn = {0035-8711},
    doi = {10.1093/mnras/staa1365},
    url = {https://doi.org/10.1093/mnras/staa1365},
    note = {staa1365},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/staa1365/33220687/staa1365.pdf},
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
