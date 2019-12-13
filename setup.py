#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages


setup(
    name="frankenstein",
    version="0.1.0",
    packages=find_packages(),
    author="Jeff Jennings",
    author_email="jmj51@ast.cam.ac.uk",
    description="Frankenstein desc",
    long_description=open('README.rst').read(),
    install_requires=[line.rstrip() for line in open("requirements.txt", "r").readlines()],
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
