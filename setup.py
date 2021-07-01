from setuptools import Extension, setup

from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('inkid.data.Volume', ['inkid/data/Volume.pyx'],
              include_dirs=[np.get_include()]),
]

setup(
    name='inkid',
    version='0.0.1',
    description='Identify ink via machine learning.',
    url='https://code.vis.uky.edu/seales-research/ink-id',
    author='University of Kentucky',
    license='GPLv3',
    packages=['inkid'],
    install_requires=[
        'autopep8',
        'configargparse',
        'Cython',
        'dicttoxml',
        'gitpython',
        'humanize',
        'imageio',
        'jsmin',
        'kornia',
        'mathutils',
        'matplotlib',
        'Pillow',
        'pygifsicle',
        'pylint',
        'pywavelets',
        'scikit-learn',
        'sphinx',
        'tensorboard',
        'torch==1.6.0',
        'torch-summary',
        'tqdm',
        'wand',
    ],
    ext_modules=cythonize(extensions, annotate=True),
    entry_points={
        'console_scripts': [
            'inkid-train-and-predict = scripts.train_and_predict:main',
            'inkid-summary = scripts.misc.create_summary_images:main',
            'inkid-rclone-upload = scripts.misc.rclone_upload:main',
        ],
    },
    zip_safe=False,
)
