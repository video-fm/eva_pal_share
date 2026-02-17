#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="eva",
    version="0.1.4",
    description="Eva Franka Infrastructure - A modular robotics framework built on DROID",
    author="GRASP Lab",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
    ],
    # python_requires=">=3.8",
)
