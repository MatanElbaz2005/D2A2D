# setup.py
from setuptools import setup, Extension
import sys
import os

try:
    import pybind11
except ImportError:
    raise RuntimeError("pybind11 is required at build time. Install it first: pip install pybind11")

extra_compile_args = ["-O3", "-fexceptions", "-std=c++17"]
# On ARM (Raspberry Pi 4/5) this enables NEON/POPCNT:
extra_compile_args += ["-march=native"]

ext_modules = [
    Extension(
        name="binxcorr",
        sources=["binxcorr.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="binxcorr",
    version="0.1.0",
    description="Binary sliding correlation (XOR+POPCNT) with a pybind11 binding",
    ext_modules=ext_modules,
    zip_safe=False,
)
