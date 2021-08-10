from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'inkid.data.volume',
        ['inkid/data/volume.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'inkid.data.mathutils',
        ['inkid/data/mathutils.pyx']
    )
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
)
