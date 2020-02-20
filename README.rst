========
 ink-id
========

``inkid`` is a Python package and collection of scripts for identifying ink in a document via machine learning.

Requirements
============

Python >=3.7 is required.

Installation
============

.. code-block:: bash

    $ git clone https://code.cs.uky.edu/seales-research/ink-id.git && cd ink-id
    $ pip3 install -U virtualenv            # Install virtualenv
    $ virtualenv -p python3 env-inkid       # Create a new environment named env-inkid
    $ source env-inkid/bin/activate         # Activate the environment
    (env-inkid) $ pip install Cython numpy  # Install prerequisites
    (env-inkid) $ pip install -e .          # Install ink-id and dependencies
    (env-inkid) $ deactivate                # When finished, deactivate the environment

Usage
=====

The package can be used as a Python library:

.. code-block:: python

   import inkid

   params = inkid.ops.load_default_parameters()
   regions = inkid.data.RegionSet.from_json(region_set_filename)

There are also some applications included, for example:

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
<https://code.vis.uky.edu/seales-research/ink-id/blob/develop/scripts/train_and_predict.py>`_ typically takes a region set file as input and trains on the specified training regions, validates on the validation regions, and predicts on the prediction regions. However if the ``-k`` argument is passed, the behavior is slightly different. In this case it expects the input region set to have only a set of training regions, with validation and prediction being empty. The kth training region will be removed from the training set and added to the validation and prediction sets. Example:

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

When contributing to this repository, please first discuss the change you wish to make via issue, email, or another method with the owners of this repository.

We follow the git branching model described `here <http://nvie.com/posts/a-successful-git-branching-model/>`_
and document code based on the `Google Python Style Guide standards <https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_.

License
=======

This package is licensed under the Microsoft Reference Source License (MS-RSL) - see `LICENSE <https://code.cs.uky.edu/seales-research/ink-id/blob/develop/LICENSE>`_ for details.
