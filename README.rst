ink-id
======

``ink-id`` is a Python package and collection of scripts for identifying ink in a document via machine learning.

Requirements
------------

Python >=3.4 is supported by this package.

Python 2 is not supported.

Installation
------------

First, clone this repository and navigate to the root directory:

.. code-block:: bash

   $ git clone https://code.vis.uky.edu/seales-research/ink-id.git
   $ cd ink-id

To install the package you can use `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: bash

   $ pip install -e .

This will install the ``inkid`` package from the current directory, based on the contents of ``setup.py``.

You can also (recommended) use `pipenv <https://docs.pipenv.org/>`_ to create a virtual environment and install the package to that environment:

.. code-block:: bash

   $ pipenv --three       # Create a new virtual environment with Python 3
   $ pipenv install -e .  # Install the inkid package to the virtual environment, using symlink so that changes to the source are reflected in the installation

Some other useful pipenv commands:

.. code-block:: bash

   $ pipenv graph   # View the installed packages to confirm inkid and dependencies are installed
   $ pipenv shell   # Enter the created virtual environment containing the inkid installation
   $ pipenv update  # Uninstall all packages and reinstall. Useful after certain changes, like adding a console script

Documentation
-------------

TODO. Will use Sphinx.

Usage
-----

The package can be imported into Python programs, for example:

.. code-block:: python

   import inkid.volumes
   import inkid.ops

   params = inkid.ops.load_default_parameters()
   volumes = inkid.volumes.VolumeSet(params)

There are also some console scripts included, for example:

::

   $ inkid-top-n -h
   usage: inkid-top-n [-h] --data path --surfacemask path --surfacedata path
                      --model path [--number N] [--outputdir path]

   Using a trained model and volume data, get the n subvolumes on that surface
   that have the highest and lowest prediction score in the model.

   optional arguments:
       -h, --help            show this help message and exit
       --data path, -d path  path to volume data (slices directory)
       --surfacemask path    path to surface mask image
       --surfacedata path    path to surface data
       --model path          path to trained model
       --number N, -n N      number of subvolumes to keep
       --outputdir path      path to output directory


Contributing
------------

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository.

Git branching model
~~~~~~~~~~~~~~~~~~~

We follow the development model described `here <http://nvie.com/posts/a-successful-git-branching-model/>`_. Anything in the ``master`` branch is considered production. Most work happens in a feature branch that is merged into ``develop`` before being merged into ``master``.

Documenting
~~~~~~~~~~~

Please document code (notably functions and classes) using doc strings according to the `Google Python Style Guide standards <https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_. This will ensure that your notes are automatically picked up and included in the generated documentation.

Console Scripts
~~~~~~~~~~~~~~~

New console/command line scripts can be added to the package using the ``entry_points['console_scripts']`` array in ``setup.py``.

License
-------

This package is licensed under the Microsoft Reference Source License (MS-RSL) - see `LICENSE <https://code.vis.uky.edu/seales-research/ink-id/blob/develop/LICENSE>`_ for details.
