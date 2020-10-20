from setuptools import Extension, setup

from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('inkid.data.Volume', ['inkid/data/Volume.c'],
              include_dirs=[np.get_include()]),
]

setup(
    name='inkid',
    version='0.0.1',
    description='Identify ink via machine learning.',
    url='https://code.vis.uky.edu/seales-research/ink-id',
    author='University of Kentucky',
    license='MS-RSL',
    packages=['inkid'],
    install_requires=[
        'autopep8',
        'configargparse',
        'Cython',
        'gitpython',
        'imageio',
        'jsmin',
        'kornia',
        'mathutils',
        'matplotlib',
        'Pillow',
        'progressbar2',
        'pylint',
        'pywavelets',
        'scikit-learn',
        'sphinx',
        'tensorboard',
        'torch==1.5.0',
        'torch-summary',
        'wand',
    ],
    ext_modules=cythonize(extensions, annotate=True),
    entry_points={
        'console_scripts': [
            'inkid-train-and-predict = scripts.train_and_predict:main',
            'inkid-summary = scripts.misc.add_k_fold_prediction_images:main',
        ],
    },
    zip_safe=False,
)
