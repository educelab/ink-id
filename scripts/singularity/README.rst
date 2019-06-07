=====================
Singularity Container
=====================
This directory contains the scripts and information needed for building and
running Singularity containers for ``ink-id``.

Additional information on Singularity is found at `singularity 
<https://www.sylabs.io/guides/2.5/user-guide/#>`_.

Singularity Installation
========================
To build and run Singularity containers, Singularity must be installed first.
Follow the installation guide found here: 
`Singularity Installation <https://www.sylabs.io/guides/2.5/user-guide/quick_start.html#quick-installation-steps>`_.

How to Build ink-id Containers Using Def Files
==============================================

Build a Singularity Container for CPU
-------------------------------------
1. Create a text file containing at least two lines: gitlab username on the first
   line and password on the second line and save it locally. Third line can be
   added if a branch other than master is to be used when running the container.
   The file should contain nothing but the username, password, and, optionally,
   the branch name in each line. 

2. Run the following command (.sif file name does not have to match the .def
   file) to build a Singularity container. 

   .. code-block:: bash
   
      $ sudo singularity build inkid-cpu.sif inkid-cpu.def


3. The freshly built Singularity container (.sif) already has all the ``ink-id``
   software and all the dependencies installed inside. 

Build a Singularity Container for GPU (specifically for LCC)
------------------------------------------------------------
1. As mentioned above, edit the ``username`` and ``password`` in the .def file.
2. Since this container needs to have various GPU related files, packages, and 
   software, run the following command to download the ``tf-gpu`` image in the 
   same directory where the ``inkid-gpu.def`` file is stored:

   .. code-block:: bash
   
       $ wget 'http://mirror.ccs.uky.edu/singularity/TensorFlow/tf-gpu-1.13.1.sif'
   

3. Build the container:
 
   .. code-block:: bash
   
       $ sudo singularity build inkid-gpu.sif  inkid-gpu.def
   
4. Copy the ``.sif`` file to LCC.

Using the Singularity Container
===============================
For detailed instruction, refer to the Singularity Doc mentioned above.

On all machines, the two most useful commands are

.. code-block:: bash
   
   $ singularity run inkid-cpu.sif inkid-train-and predict <args>
   $ singularity run python <python_script>.py <args>

To use singularity shell, remember to activate the virtualenv environment first

.. code-block:: bash

   $ singularity shell
   $ . /tensorflow/bin/activate
   
For LCC, remember to load the singularity module and pass the ``--nv`` parameter

.. code-block:: bash

   $ singularity run --nv inkid-gpu.sif inkid-train-and-predict <args>

Viewing Definition File Inside the Container
============================================
Given ``<container>.sif`` file, ``<container>.def`` file that was used to build
the container can be viewed with the following command.  This is particularly a 
useful feature when pulling a container image from the container library.

.. code-block:: bash

   $ singulairy exec <container>.sif cat /.singularity.d/Singularity

   
Slurm Script for Running Jobs on LCC at University of Kentucky
==============================================================
``submit.sh`` is a slurm script template for running a job on LCC using the ink-id 
Singularity container.  It is important to specify the size of memory, upper limit
on the running time as the project would be billedd to the capacity of the machine,
regardless of the actual resource usage, if those parameters are unspecified.

``submit_example1.sh`` is the actual script that was used to run a job on LCC in 
April 2019.

