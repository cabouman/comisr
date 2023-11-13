from setuptools import setup, find_packages, Extension
import numpy as np
import os

NAME = "comiser"
VERSION = "0.1"
DESCR = "package for video super-resolution"
REQUIRES = ['numpy']
LICENSE = "BSD-3-Clause"

AUTHOR = 'comiser development team'
EMAIL = "buzzard@purdue.edu"
PACKAGE_DIR = "comiser"

setup(install_requires=REQUIRES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(include=['comiser']),
      )

