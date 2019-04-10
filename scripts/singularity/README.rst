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
1. Replace the ``username`` and ``password`` with your own and save the file . 
   This pulls the latest version of ``ink-id`` code from the repo.

2. Run the following command (.sif file name does not have to match the .def
   file) to build a Singularity container. 

   .. code-block:: bash
   
      $ git clone https://code.cs.uky.edu/seales-research/ink-id.git
      $ cd ink-id


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
   
Slurm Script
============
``submit.sh`` is a sample script for submitting a job on LCC using ink-id 
Singularity container.

