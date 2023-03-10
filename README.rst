========
 ink-id
========

``inkid`` is a Python package and collection of scripts for identifying ink in volumetric CT data using machine learning.

Installation
============

``inkid`` is available `on PyPI <https://pypi.org/project/inkid/>`_ and can be installed via ``pip``:

.. code-block:: bash

    $ pip install inkid

To install the source for development:

.. code-block:: bash

    $ git clone https://github.com/educelab/ink-id.git && cd ink-id  # Clone the repository
    $ python -m venv .venv  # Create a virtual environment
    $ . .venv/bin/activate  # Activate the environment
    (.venv) $ pip install -U pip  # Upgrade pip
    (.venv) $ pip install -e .[dev]  # Install ink-id and dependencies, including those specific to development
    (.venv) $ deactivate  # When finished, deactivate the environment

Cython modules are automatically built during the installation.
After changes to Cython files (``.pyx`` and ``.pxd``), those modules must be rebuilt:

.. code-block:: bash

    $ python setup.py build_ext --inplace

To set up Weights & Biases for experiment tracking, go to `wandb.ai/settings <https://wandb.ai/settings>`_ to copy your API key, and then run:

.. code-block:: bash

    $ wandb login

Use `--wandb` with `inkid-train-and-predict` to log job results to W&B. 

Requirements
============

Python >=3.8 is required.

Usage
=====

The package can be used as a Python library:

.. code-block:: python

   import inkid

   params = inkid.util.json_schema('dataSource0.1')
   regions = inkid.data.Dataset([os.path.join(inkid.util.dummy_volpkg_path(), 'working', 'DummyTest_grid1x2.txt')])

A script is also included for running a training job and/or generating prediction images:

::

   $ inkid-train-and-predict
   usage: inkid-train-and-predict [options]

Examples
--------

SLURM Jobs
^^^^^^^^^^

``inkid`` is often run on a compute cluster, scheduled by SLURM and run in a Singularity container.
For documentation of this usage, see ``singularity/inkid.def``.

K-Fold Cross Validation (and Prediction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``inkid/scripts/train_and_predict.py`` typically takes dataset files as input and trains on the specified training
regions, validates on the validation regions, and predicts on the prediction regions.
However if the ``--cross-validate-on`` argument is passed, the behavior is slightly different.
The nth training region will be removed from the training set and added to the validation and prediction sets. Example:

.. code-block:: bash

   $ inkid-train-and-predict \
       --training-set inkid/examples/DummyTest.volpkg/working/DummyTest_grid1x2.txt \
       --output test \
       --cross-validate-on 0

It is possible to schedule all of the k-fold jobs with one command if using SLURM's ``sbatch`` via the ``--array``
argument. ``submit.sh`` creates a job for each array value, passing that value automatically to ``--cross-validate-on``:

.. code-block:: bash

   $ ./submit_with_summary.sh sbatch -p P4V12_SKY32M192_L --time=1-00:00:00 --mem=187G --array=0-1 submit.sh \
        --training-set /pscratch/seales_uksr/dri-datasets-drive/Dummy/DummyTest.volpkg/working/DummyTest_1x2Grid.txt \
        --subvolume-shape-voxels 48 48 48 \
        --final-prediction-on-all \
        --prediction-grid-spacing 2 \
        --label-type rgb_values \
        --subvolume-shape-microns 300 20 20 \
        --output /pscratch/seales_uksr/dri-experiments-drive/inkid/results/DummyTest/check_gpu/03

After performing a run for each value of ``--cross-validate-on``, each will have created a subdirectory of output.

Generating Summary Images
^^^^^^^^^^^^^^^^^^^^^^^^^

There is a script ``inkid/scripts/create_summary_images.py`` that takes the parent output directory and will
generate various output images combining the cross-validation results. Example:

.. code-block:: bash

   $ python inkid/scripts/create_summary_images.py ~/data/out/carbon_phantom_col1_test/

Grid Training
^^^^^^^^^^^^^

When working with only one surface PPM, it is often desirable to split that single region into a grid to be used with
k-fold cross-validation. There is a script to automatically create the grid dataset file:

.. code-block:: bash

   $ python inkid/scripts/split_region_into_grid.py inkid/examples/DummyTest.volpkg/working/DummyTest.json 1 2

Then use this dataset for standard k-fold cross validation and prediction.

Miscellaneous
^^^^^^^^^^^^^

There is a dummy test dataset in the DRI Datasets Drive that is meant to be a small volume to quickly validate
training and prediction code. If something major has been broken such as dimensions in the neural network model, this
will make that clear without having to wait for large volumes to load. Example:

.. code-block:: bash

   $ ./submit_with_summary.sh sbatch -p P4V12_SKY32M192_L --time=1-00:00:00 --mem=150G submit.sh \
        --training-set $PSCRATCH/seales_uksr/dri-datasets-drive/Dummy/DummyTest.volpkg/working/DummyTest_grid1x2.txt \
        --subvolume-shape-voxels 48 2 2 \
        --final-prediction-on-all \
        --prediction-grid-spacing 2 \
        --label-type rgb_values \
        --cross-validate-on 0 \
        --output ~/temp/test00

Texture a region using an existing trained model (important parts: ``--model`` and ``--skip-training``:

.. code-block:: bash

   $ ./submit_with_summary.sh sbatch -p P4V12_SKY32M192_L --time=1-00:00:00 --mem=187G submit.sh \
        --training-set $PSCRATCH/seales_uksr/dri-datasets-drive/MorganM910/MS910.volpkg/working/segmentation/quire.json \
        --prediction-set $PSCRATCH/seales_uksr/dri-datasets-drive/MorganM910/MS910.volpkg/working/segmentation/p60.json \
        --subvolume-shape 48 48 48 \
        --final-prediction-on-all \
        --prediction-grid-spacing 8 \
        --label-type rgb_values \
        --skip-training \
        --model $PSCRATCH/seales_uksr/dri-experiments-drive/inkid/results/MS910/p60/initial/09/2021-02-08_09.15.07/checkpoints/checkpoint_0_175000.pt \
        --output $PSCRATCH/seales_uksr/dri-experiments-drive/inkid/results/MS910/p60/fromSavedWeights/02

Contributing
============

When contributing to this repository, please first discuss the change you wish to make via issue, email, or another method with the owners of this repository.

We follow the git branching model described `here <http://nvie.com/posts/a-successful-git-branching-model/>`_.

License
=======

This package is licensed under the GNU General Public License (GPLv3) - see ``LICENSE`` for details.

Citation
============

If you use ``inkid`` in your research, please cite the following publication:

.. code-block:: bibtex

    @article{parker2019invisibility,
       title={From invisibility to readability: recovering the ink of Herculaneum},
       author={Parker, Clifford Seth and Parsons, Stephen and Bandy, Jack and Chapman, Christy and Coppens, Frederik and Seales, William Brent},
       journal={PloS one},
       volume={14},
       number={5},
       pages={e0215775},
       year={2019},
       publisher={Public Library of Science}
    }

