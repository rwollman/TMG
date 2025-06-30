from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ConnectedComponentEntropy",
        ["ConnectedComponentEntropy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="ConnectedComponentEntropy",
    ext_modules=cythonize(extensions),
) 