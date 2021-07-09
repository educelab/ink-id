from setuptools import Extension, setup, find_packages

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
    packages=find_packages(include=['inkid', 'inkid.*']),
    install_requires=[
        'autopep8',
        'configargparse',
        'Cython',
        'dicttoxml',
        'gitpython',
        'humanize',
        'imageio',
        'jsmin',
        'mathutils',
        'matplotlib',
        'Pillow==8.2.0',  # Temporary due to bug in 8.3.0 https://github.com/pytorch/pytorch/issues/61125
        'pygifsicle',
        'pylint',
        'pywavelets',
        'scikit-learn',
        'sphinx',
        'tensorboard',
        'torch',
        'torch-summary',
        'tqdm',
        'wand',
    ],
    ext_modules=cythonize(extensions, annotate=True),
    entry_points={
        'console_scripts': [
            'inkid-train-and-predict = inkid.scripts.train_and_predict:main',
            'inkid-summary = inkid.scripts.create_summary_images:main',
            'inkid-rclone-upload = inkid.scripts.rclone_upload:main',
        ],
    },
    zip_safe=False,
)
