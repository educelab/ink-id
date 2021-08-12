from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'inkid.data.volume',
        ['inkid/data/volume.pyx'],
        include_dirs=[np.get_include()],
        # https://cython.readthedocs.io/en/latest/src/userguide/migrating_to_cy30.html?highlight=deprecated#numpy-c-api
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        'inkid.data.mathutils',
        ['inkid/data/mathutils.pyx']
    )
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
)
