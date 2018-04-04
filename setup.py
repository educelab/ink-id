from setuptools import setup

setup(name='inkid',
      version='0.0.1',
      description='Identify ink via machine learning.',
      url='https://code.vis.uky.edu/seales-research/ink-id',
      author='University of Kentucky',
      license='MS-RSL',
      packages=['inkid'],
      install_requires=[
          'tensorflow-gpu',
          'jsmin',
          'matplotlib',
          'Pillow',
          'sklearn',
          'imageio',
      ],
      entry_points = {
          'console_scripts': [
              'inkid-train-and-predict = scripts.train_and_predict:main',
              'inkid-top-n = scripts.get_top_bottom_n_subvolumes:main',
          ],
      },
)

