<p align="center">
  <img width = "800" src="https://github.com/discsim/frank/blob/master/docs/images/day_off.png?raw=true"/>
</p>

<p align="center">
  <a href="https://github.com/discsim/frank/releases">
      <img src="https://img.shields.io/github/release/discsim/frank/all.svg">
  </a>

  <a href="https://pypi.python.org/pypi/frank">
      <img src="https://img.shields.io/pypi/v/frank.svg">
  </a>

  <a href="https://discsim.github.io/frank/">
    <img src="https://img.shields.io/badge/docs-Read%20em!-blue.svg?style=flat"/>

  <br/>
  <a href="https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staa1365/5838058?guestAccessKey=7f163a1f-c12f-4771-8e54-928636794a5b">
      <img src="https://img.shields.io/badge/paper-MNRAS-blue.svg">
  </a>

  <a href="https://doi.org/10.5281/zenodo.3832065"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3832065.svg" alt="DOI">
  </a>


  <br/>
  <a href="https://circleci.com/gh/discsim/frank">
      <img src="https://circleci.com/gh/discsim/frank.svg?style=shield">
  </a>    

  <a href="https://discsim.github.io/frank/coverage/index.html">
      <img src="https://discsim.github.io/frank/coverage/badge.svg">
  </a>   

  <br/>
  <a href="https://www.gnu.org/licenses/lgpl-3.0">
      <img src="https://img.shields.io/badge/License-LGPL%20v3-blue.svg"
      [![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg">
  </a>      
</p>

Frankenstein (**frank**) is a library that fits the 1D radial brightness profile of an interferometric source given a set of visibilities. It uses a Gaussian process that performs the fit in <1 minute for a typical protoplanetary disc continuum dataset.

Get the code
------------
**frank**'s on [PyPI](https://pypi.org/project/frank), so you can just use pip,
```
pip install frank
```

Documentation
-------------
The [docs](https://discsim.github.io/frank/) have it all.

Attribution
-----------
If you use **frank** for your research please cite Jennings, Booth, Tazzari et al. 2020 MNRAS (accepted)
[[MNRAS]](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staa1365/5838058?guestAccessKey=7f163a1f-c12f-4771-8e54-928636794a5b) [[arXiv]](xx)
[[ADS]](xx)
[[Zenodo]](https://doi.org/10.5281/zenodo.3832065):
```
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
```

Authors
-------
- [Richard 'Dr. Frankenstein' Booth (University of Cambridge)](https://github.com/rbooth200)

- [Jeff 'The Monster' Jennings (University of Cambridge)](https://github.com/jeffjennings)

- [Marco 'It's Alive!!!' Tazzari (University of Cambridge)](https://github.com/mtazzari)

### Contact ###
Interested in collaborating to improve, extend or apply **frank**?
Or just have questions about the code that don't require submitting an issue?
[Email Jeff!](mailto:jmj51@ast.cam.ac.uk)

License
-------
**frank** is free software licensed under the LGPLv3 License. For more details see the [LICENSE](https://github.com/discsim/frank/blob/master/LICENSE.txt).

© Copyright 2019-2020 Richard Booth, Jeff Jennings, Marco Tazzari.

Image: Universal Studios, NBCUniversal [Public domain]
