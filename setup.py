from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('inkid.data.volume', ['inkid/data/volume.pyx'],
              include_dirs=[np.get_include()]),
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
)
