#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

version = {}
exec(open("frank/__init__.py", "r").read(), version)
version = version['__version__']

setup(
    name="frank",
    version=version,
    packages=find_packages(),
    include_package_data=True,
    author="Richard Booth, Jeff Jennings, Marco Tazzari",
    author_email="jmj51@ast.cam.ac.uk",
    description="Frankenstein, the flux reconstructor",
    long_description=open('README.md').read(),
    python_requires='>=3',
    install_requires=[line.rstrip() for line in open("requirements.txt", "r").readlines()],
    extras_require={
        'test' : ['pytest'],
        'docs-build' : ['sphinx', 'sphinxcontrib-fulltoc', 'sphinx_rtd_theme', 'nbsphinx'],
        },
    license="GPLv3",
    url="https://github.com/discsim/frank",
    keywords=["science", "astronomy", "interferometry"]
    classifiers=[
         "Development Status :: 1 - Production/Stable",
         "Intended Audience :: Developers",
         "Intended Audience :: Science/Research",
         "Operating System :: OS Independent",      
         "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
         "Programming Language :: Python :: 3",
    ]
)
