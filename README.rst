ink-id
======

``inkid`` is a Python package and collection of scripts for identifying ink in a document via machine learning.

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

To install the package you can use `pipenv <https://docs.pipenv.org/>`_:

.. code-block:: bash

   $ pip install pipenv   # If needed, install pipenv
   $ pipenv --three       # Create a new virtual environment with Python 3
   $ pipenv install -e .  # Install inkid to the virtual environment and use symlinks

The install command will find ``setup.py`` and install the dependencies for ``inkid``.

The default installation assumes you have already installed ``tensorflow`` on your machine. If you wish to install ``tensorflow`` along with ``inkid``, you can run either of these commands depending on whether or not you wish to include GPU support:

.. code-block:: bash

   $ pipenv install -e .[tf]      # Install inkid and install tensorflow (CPU only)
   $ pipenv install -e .[tf_gpu]  # Install inkid and install tensorflow-gpu

Some other useful pipenv commands:

.. code-block:: bash

   $ pipenv graph   # View the installed packages to confirm inkid and dependencies are installed
   $ pipenv shell   # Enter the created virtual environment containing the inkid installation
   $ pipenv update  # Uninstall all packages and reinstall. Useful after certain changes, like adding a console script
   
Pipenv is recommended, but you could also just use pip:

.. code-block:: bash
   
   $ pip install -e .

Installation on IBM Power8 Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since a global install is not possible, install locally:

.. code-block:: bash

   $ pip3 install -e . --user --upgrade

Tensorboard on IBM Power8 Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to ssh into the server with port forwarding so that you can view the Tensorboard output on your local machine. To do so check out this `answer <https://stackoverflow.com/a/40413202>`_.

Grid Training
-------------

To perform grid training, create a RegionSet JSON file for the PPM
with only one training region (with no bounds, meaning it will default
to the full size of the PPM). Then use
``scripts/misc/split_region_into_grid.py`` to split this into a grid
of the desired shape. Then use ``scripts/train_and_predict.py`` with
the ``-k`` argument to indicate which grid square should be isolated
for evaluation and prediction.

Documentation
-------------

TODO. Will use Sphinx.

Usage
-----

The package can be imported into Python programs, for example:

.. code-block:: python

   import inkid.volumes

   params = inkid.ops.load_default_parameters()
   regions = inkid.data.RegionSet.from_json(region_set_filename)

There are also some console scripts included, for example:

::

   $ inkid-train-and-predict
   usage: inkid-train-and-predict [-h] -d path [-o path] [-m path] [-k num]
   
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
