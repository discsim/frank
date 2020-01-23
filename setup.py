#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

version = {}
exec(open("frankenstein/__init__.py", "r").read(), version)
version = version['__version__']

setup(
    name="frankenstein",
    version=version,
    packages=find_packages(),
    author="Jeff Jennings",
    author_email="jmj51@ast.cam.ac.uk",
    description="Frankenstein desc",
    long_description=open('README.rst').read(),
    install_requires=[line.rstrip() for line in open("requirements.txt", "r").readlines()],
    extras_require={
        'test' : ['pytest'],
        'docs-build' : ['sphinx', 'sphinxcontrib-fulltoc'],
        },
    license="tbd",
    url="tbd",
    classifiers=[
        # 'Development Status :: 1 - Production/Stable',
        # "Intended Audience :: Developers",
        # "Intended Audience :: Science/Research",
        # 'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3',
    ]
)
