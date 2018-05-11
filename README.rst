========
 ink-id
========

``inkid`` is a Python package and collection of scripts for identifying ink in a document via machine learning.

Requirements
============

Python >=3.4 is supported by this package.

Python 2 is not supported.

Installation
============

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
---------------------------------

Since a global install is not possible, install locally:

.. code-block:: bash

   $ pip3 install -e . --user --upgrade

Tensorboard on IBM Power8 Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to ssh into the server with port forwarding so that you can view the Tensorboard output on your local machine. To do so check out this `answer <https://stackoverflow.com/a/40413202>`_.

Documentation
=============

TODO. Will use Sphinx.

Usage
=====

The package can be imported into Python programs, for example:

.. code-block:: python

   import inkid.volumes

   params = inkid.ops.load_default_parameters()
   regions = inkid.data.RegionSet.from_json(region_set_filename)

There are also some console scripts included, for example:

::

   $ inkid-train-and-predict
   usage: inkid-train-and-predict [-h] -d path [-o path] [-m path] [-k num]

Examples
--------

Grid Training
~~~~~~~~~~~~~

To perform grid training, create a RegionSet JSON file for the PPM with only one training region (with no bounds, meaning it will default to the full size of the PPM). For example:
`examples/region-set-files/lunate-sigma-one-region.json <https://code.vis.uky.edu/seales-research/ink-id/blob/develop/examples/region_set_files/lunate-sigma-one-region.json>`_.

Then use `scripts/misc/split_region_into_grid.py <https://code.vis.uky.edu/seales-research/ink-id/blob/develop/scripts/misc/split_region_into_grid.py>`_ to split this into a grid of the desired shape. Example:

.. code-block:: bash

   $ python scripts/misc/split_region_into_grid.py \
		-i ~/data/lunate-sigma/lunate-sigma.json \
		-o lunate-sigma-grid-2x5.json \
		-columns 2 \
		-rows 5

Then use this region set for standard k-fold cross validation and prediction.

K-Fold Cross Validation (and Prediction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
`scripts/train_and_predict.py
<https://code.vis.uky.edu/seales-research/ink-id/blob/develop/scripts/train_and_predict.py>`_ typically takes a region set file as input and trains on the specified training regions, evaluates on the evaluation regions, and predicts on the prediction regions. However if the ``-k`` argument is passed, the behavior is slightly different. In this case it expects the input region set to have only a set of training regions, with evaluation and prediction being empty. The kth training region will be removed from the training set and added to the evaluation and prediction sets. Example:

.. code-block:: bash

   $ inkid-train-and-predict -d ~/data/lunate-sigma/grid-2x5.json -o ~/data/out/ -k 7 --final-prediction-on-all

It is possible to run all of these with one command if using ``sbatch`` on the server. Example:

.. code-block:: bash

   $ sbatch --array=0-4%2 scripts/slurm_train_and_predict.sh -d ~/data/CarbonPhantomV3.volpkg/working/2/Col2_k-fold-characters-region-set.json -o ~/data/out/col2_not_flattened --final-prediction-on-all

After performing a run for each value of k, each will have created a directory of output. If these are all in the same parent directory, there is a script to merge together the individual predictions into a final prediction image. If ``--best-f1`` is passed, it will take the prediction with the best f1 score for each individual region, rather than the final prediction for that region. Example:

.. code-block:: bash

   $ python scripts/misc/add_k_fold_prediction_images.py --dir ~/data/out/carbon_phantom_col1_test/ --outfile added_image.tif --best-f1

Contributing
============

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository.

Git branching model
-------------------

We follow the development model described `here <http://nvie.com/posts/a-successful-git-branching-model/>`_. Anything in the ``master`` branch is considered production. Most work happens in a feature branch that is merged into ``develop`` before being merged into ``master``.

Documenting
-----------

Please document code (notably functions and classes) using doc strings according to the `Google Python Style Guide standards <https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_. This will ensure that your notes are automatically picked up and included in the generated documentation.

Console Scripts
---------------

New console/command line scripts can be added to the package using the ``entry_points['console_scripts']`` array in ``setup.py``.

License
=======

This package is licensed under the Microsoft Reference Source License (MS-RSL) - see `LICENSE <https://code.vis.uky.edu/seales-research/ink-id/blob/develop/LICENSE>`_ for details.
