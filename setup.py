from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("inkid.data.cythonutils", ["inkid/data/cythonutils.pyx"])
]

setup(
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
