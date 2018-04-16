from setuptools import setup

setup(name='inkid',
      version='0.0.1',
      description='Identify ink via machine learning.',
      url='https://code.vis.uky.edu/seales-research/ink-id',
      author='University of Kentucky',
      license='MS-RSL',
      packages=['inkid'],
      install_requires=[
          'imageio',
          'jsmin',
          'mathutils',
          'matplotlib',
          'Pillow',
          'progressbar2',
      ],
      entry_points = {
          'console_scripts': [
              'inkid-train-and-predict = scripts.train_and_predict:main',
          ],
      },
      extras_require = {
          'tf': ['tensorflow>=1.5.0'],
          'tf_gpu': ['tensorflow-gpu>=1.5.0'],
      },
)

