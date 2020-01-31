# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../'))

import frankenstein

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'frankenstein'
authors = u'R. Booth, J. Jennings, M. Tazzari.'
copyright = '2019-%d, %s' % (datetime.now().year, authors)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = str(frankenstein.__version__)
# The full version, including alpha/beta/rc tags.
release = str(frankenstein.__version__)

# Get current git branch
branch = os.getenv("GHBRANCH", "master")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'matplotlib.sphinxext.plot_directive',
    #'sphinxcontrib.fulltoc',
]

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

master_doc = 'index'

# A string of reStructuredText that will be included at the beginning of every
# source file that is read.
rst_prolog="""
.. |frank| replace:: **{}**
.. default-role:: code
""".format(project)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'display_version': True}
html_last_updated_fmt = '%Y %b %d at %H:%M:%S UTC'
html_show_sourcelink = False
html_logo = 'images/prom_photo.jpg'
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension settings ------------------------------------------------------

# autodocs
autoclass_content = 'both'
autosummary_generate = True
autodoc_docstring_signature = True

# Add a heading to notebooks
nbsphinx_prolog = """
{%s set docname = env.doc2path(env.docname, base=None) %s}
.. note:: This tutorial is produced by the Jupyter notebook
`here <https://github.com/discsim/frankenstein/blob/%s/{{ docname }}>`_.
""" % ("%", "%", branch,)

# nbsphinx
nbsphinx_prompt_width = 0
nbsphinx_timeout = 600
napoleon_use_ivar = True
nbsphinx_allow_errors = True # TODO: remove once notebooks final
