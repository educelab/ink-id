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
The basic installation steps can be summarized as follows:
1. Clone this repository and navigate to its root directory.
2. (Optional but highly recommended) Create a virtual environment.
3. Install Cython and numpy.
4. (Optional) Install sphinx, pylint, and autopep8.  
4. Install tensorflow(standard or GPU) and ink-id, which will install additional
packages.

Installation Using Singularity Container
----------------------------------------
See the documentation under the Singularity directory. (TODO)

Installation Methods Using Pipenv
---------------------------------
Official doc: `pipenv <https://docs.pipenv.org/>`_.
Intro to pipenv: `pipenvguide <https://realpython.com/pipenv-guide/>`_. 

First, clone this repository and navigate to the root directory:

.. code-block:: bash

   $ git clone https://code.cs.uky.edu/seales-research/ink-id.git
   $ cd ink-id

Install pipenv:

.. code-block:: bash

   $ pip install pipenv

Creating Pipenv Environment from Scratch for Developing/Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use this method if there is no Pipfile/Pipfile.lock. 
.. code-block:: bash

   $ pipenv --three               # Create a new virtual environment with Python 3
   $ pipenv install Cython numpy  # These are required before inkid can be installed
   $ pipenv install sphinx pylint autopep8 --dev  # Developing tools (optional)

At this point, the newly created Pipfile should list all the installed packages,
which are all the packages that need to be installed before inkid and tensorflow
can be installed. At this point, activate the pipenv environment and install 
tensorflow(-gpu) and ink-id

.. code-block:: bash

   $ pipenv shell                   # Activate the virtual environment shell
   (ink-id) $ pip install -e .[tf]  # Inside the environment, install tensorflow
                                      and ink-id

The last command will find ``setup.py`` and install the dependencies for ``inkid``.
If tensorflow is already installed, do not include ``[tf]``. To install 
``tensorflow-gpu`` use instead

.. code-block:: bash

   $ pipenv install -e .[tf_gpu]  # Install inkid and install tensorflow-gpu

Note: since ``tensorflow(-gpu)`` and ``inkid`` was installed using ``pip`` rather than
``pipenv``, their installation is NOT included in the Pipfile(.lock). (See the
next sections for its implication.) If the virtual environment worked, it is 
recommended that both ``Pipfile/Pipfile.lock`` be stored in the VCS repo along with
the rest of the code. This will serve as a record of an environment in which a 
particular commit was developed. To save ``Pipfile/Pipfile.lock`` for later, 
run 

.. code-block:: bash

   $ pipenv lock    


Creating Pipenv Environment from (previously created) Pipfile/Pipfile.lock
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use this method if you wish to create the same developing environment you had
when the Pipfile(.lock) was created.

.. code-block:: bash

   $ pipenv sync --three           # Create a new virtual environment in which
                                   # all the packages in Pipfile will automatically
                                   # be installed

By defaut, this installs the same versions of the packages that were locked 
(specified) in Pipfile.lock. To install the newest versions of packages, 
use ``--skip-lock``.  

At this point, follow the instruction above to install a desired type of 
tensorflow and ink-id inside the virtual environment.

Producing Pipfile for Production/Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section outlines the steps to produce a Pipfile that installs not only 
the prerequisite packages, but also tensorflow(-gpu) and ink-id.

.. code-block:: bash

   $ pipenv --three               # Create a new virtual environment with Python 3
   $ pipenv install Cython numpy  # These are required before inkid can be installed
   $ pipenv install .[tf(_gpu)]   # Install ink-id and tensorflow(-gpu)
   $ pipenv lock                  # Lock the information in Pipfile.lock

Pipfile(.lock) produced in this manner already has tensorflow(-gpu) and ink-id
installed. To recreate the virtual environment, user will need to run only

.. code-block:: bash
  
   $ pipenv sync --three          # Create a new envrionment in which
                                  # everything will automatically be installed

Some other useful pipenv commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   $ pipenv --rm    # Remove the virtualenv created for the project entirely (start all over)
   $ pipenv uninstall [package] # Uninstall a specified package. By default, it alters Pipfile.
   $ pipenv graph   # View the installed packages to confirm inkid and dependencies are installed
   $ pipenv shell   # Enter the created virtual environment containing the inkid installation
   $ pipenv update  # Uninstall all packages and reinstall. Useful after certain changes, like adding a console script
   
Installation Using Virtualenv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While ``pipenv`` can be useful for managing dependencies, lower-level ``virtualenv``
can also be used as follows.

.. code-block:: bash

    $ pip3 install -U virtualenv            # Install virtualenv
    $ virtualenv -p python3 ink-id-env      # Create a new environment named ink-id-env
    $ source ~/ink-id-env/bin/activate      # Activate the environment
    (ink-id-env) $ pip install --upgrade tensorflow(-gpu)  # Install tensorflow, which includes numpy
    (ink-id-env) $ pip install --upgrade Cython # Install Cython
    (ink-id-env) $ pip install -e .         # ink-id and its dependencies are installed
    (ink-id-env) $ pip install --upgrade sphinx pylint autopep8     # (optional)
    (ink-id-env) $ deactivate               # When finished, deactivate the environment


Installation on IBM Power8 Server
---------------------------------

Since a global install is not possible, install locally:

.. code-block:: bash

   $ pip3 install -e . --user --upgrade

Tensorboard on IBM Power8 Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to ssh into the server with port forwarding so that you can view the Tensorboard output on your local machine. To do so check out this `answer <https://stackoverflow.com/a/40413202>`_.

Documentation
=============

TODO. Will use Sphinx.

Usage
=====

The package can be imported into Python programs, for example:

.. code-block:: python

   import inkid

   params = inkid.ops.load_default_parameters()
   regions = inkid.data.RegionSet.from_json(region_set_filename)

There are also some console scripts included, for example:

::

   $ inkid-train-and-predict
   usage: inkid-train-and-predict [-h] input-file output-path [options]

Examples
--------

Grid Training
^^^^^^^^^^^^^

To perform grid training, create a RegionSet JSON file for the PPM with only one training region (with no bounds, meaning it will default to the full size of the PPM). For example:
`examples/region-set-files/lunate-sigma-one-region.json <https://code.vis.uky.edu/seales-research/ink-id/blob/develop/examples/region_set_files/lunate-sigma-one-region.json>`_.

Then use `scripts/misc/split_region_into_grid.py <https://code.vis.uky.edu/seales-research/ink-id/blob/develop/scripts/misc/split_region_into_grid.py>`_ to split this into a grid of the desired shape. Example:

.. code-block:: bash

   $ python scripts/misc/split_region_into_grid.py \
		~/data/lunate-sigma/lunate-sigma.json \
		lunate-sigma-grid-2x5.json \
		-columns 2 \
		-rows 5

Then use this region set for standard k-fold cross validation and prediction.

K-Fold Cross Validation (and Prediction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
`scripts/train_and_predict.py
<https://code.vis.uky.edu/seales-research/ink-id/blob/develop/scripts/train_and_predict.py>`_ typically takes a region set file as input and trains on the specified training regions, evaluates on the evaluation regions, and predicts on the prediction regions. However if the ``-k`` argument is passed, the behavior is slightly different. In this case it expects the input region set to have only a set of training regions, with evaluation and prediction being empty. The kth training region will be removed from the training set and added to the evaluation and prediction sets. Example:

.. code-block:: bash

   $ inkid-train-and-predict ~/data/lunate-sigma/grid-2x5.json ~/data/out/ -k 7 --final-prediction-on-all

It is possible to run all of these with one command if using ``sbatch`` on the server. Example:

.. code-block:: bash

   $ sbatch --array=0-4%2 scripts/slurm_train_and_predict.sh ~/data/CarbonPhantomV3.volpkg/working/2/Col2_k-fold-characters-region-set.json ~/data/out/col2_not_flattened --final-prediction-on-all

After performing a run for each value of k, each will have created a directory of output. If these are all in the same parent directory, there is a script to merge together the individual predictions into a final prediction image. If ``--best-f1`` is passed, it will take the prediction with the best f1 score for each individual region, rather than the final prediction for that region. Example:

.. code-block:: bash

   $ python scripts/misc/add_k_fold_prediction_images.py --dir ~/data/out/carbon_phantom_col1_test/

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

This package is licensed under the Microsoft Reference Source License (MS-RSL) - see `LICENSE <https://code.cs.uky.edu/seales-research/ink-id/blob/develop/LICENSE>`_ for details.
