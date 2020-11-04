Welcome to LambdaZero documentation
======================================

LambdaZero project explores search in the space of small molecules.

.. note:: 
   The latest version of the project can be found `here. <https://github.com/MKorablyov/LambdaZero/tree/master>`_

Installation
---------------
Install anaconda: `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_

.. code-block:: bash

   conda env create -f environment-linux.yml [-n env_name]

This will create an environment named ``lz``. LambdaZero depends on external programs (such as Dock6 and UCSF Chimera) and datasets (brutal dock and fragdb etc. ) that are not provided in this repo. These could be installed by running:

.. code-block:: bash

   bash install-prog-datasets.sh [-d dataset_path] [-p programs_path] [-s summaries_path]

this script would create a locator file called ``external_dirs.cfg`` that is machine specific and is used by the LambdaZero core to be able to call external dependencies.

Getting started
---------------------------------------
Run PPO

.. code-block:: bash

   cd ~/LambdaZero/LambdaZero/examples/PPO  
   python train_ppo.py ppo001
   
Run Ape-X

.. code-block:: bash

   cd ~/LambdaZero/LambdaZero/examples/PPO  
   python train_apex.py apex001

Run AlphaZero

.. code-block:: bash

   cd ~/LambdaZero/LambdaZero/examples/AlphaZero
   # az000 ending by three zeros means it is a debug configuration in this case it means expanding MCTS only a few times instead of 800 or 1600 times as in the original implementation to make sure the algorithm runs.
   python train_az.py az000

Train vanilla MPNN on biophysics simulation data

.. code-block:: bash

   cd ~/LambdaZero/LambdaZero/datasets/brutal_dock 
   python split_random.py
   cd ~/LambdaZero/LambdaZero/examples/mpnn
   python train_mpnn.py

Use environment, make random walks, call oracles:

.. code-block:: bash

   cd ~/LambdaZero/LambdaZero/examples/oracles
   python oracle_examples.py





.. toctree::
   :maxdepth: 1
   :caption: Get Started

   example
   

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:
   :glob:

   ./generated/LambdaZero.chem
   ./generated/LambdaZero.environments
   ./generated/LambdaZero.examples
   ./generated/LambdaZero.datasets
   ./generated/LambdaZero.inputs
   ./generated/LambdaZero.models
   ./generated/LambdaZero.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
