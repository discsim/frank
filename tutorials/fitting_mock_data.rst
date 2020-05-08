Fitting mock data
=================

The process to fit a mock dataset is the same as a real dataset apart from one potential addition.
If you've produced the mock visibilities with, e.g., a pipeline that emulates
ALMA observations, you may not have an estimate of the weights on your individual visibilities.

In this case you can set ``correct_weights=True`` in your parameter file.
This will instruct frank to use the variance of the binned visibilities to
*coarsely* estimate the pointwise weights
(see the `estimate_weights <https://github.com/discsim/frankenstein/blob/master/docs/_build/html/py_API.html#frank.utilities.estimate_weights>`_ function).
This is done after deprojecting the visibilities and before fitting for the brightness profile.
