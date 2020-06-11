.. LambdaZero documentation master file, created by
   sphinx-quickstart on Wed Jun 10 15:26:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LambdaZero documentation
==========================

Some general information about LambdaZero goes here.

* bullet 1
* bullet 2
* some other thing

more relevant text. You can create `rst` files in the source folder and link them in the
table of content (toctree).

Something important
---------------------------------------
More text.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   readme_link
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
