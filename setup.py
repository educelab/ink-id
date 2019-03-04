from setuptools import setup

from Cython.Build import cythonize
import numpy as np

setup(
    name='inkid',
    version='0.0.1',
    description='Identify ink via machine learning.',
    url='https://code.vis.uky.edu/seales-research/ink-id',
    author='University of Kentucky',
    license='MS-RSL',
    packages=['inkid'],
    install_requires=[
        'configargparse',
        'Cython',
        'gitpython',
        'imageio',
        'jsmin',
        'mathutils==2.78',
        'matplotlib',
        'Pillow',
        'progressbar2',
        'wand',
    ],
    ext_modules=cythonize('inkid/data/Volume.pyx', annotate=True),
    include_dirs=[np.get_include()],
    entry_points={
        'console_scripts': [
            'inkid-train-and-predict = scripts.train_and_predict:main',
        ],
    },
    extras_require={
        'tf': ['tensorflow>=1.5.0'],
        'tf_gpu': ['tensorflow-gpu>=1.5.0'],
    },
    zip_safe=False,
)
