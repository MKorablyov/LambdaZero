Generating  molecules in 3D
===========================

This sub-package implements experiments to predict the next block and its
attachment angle, given a parent molecule and the attachment point.

The long term goal of this work is to give the reinforcement agent control over
space geometry through the attachment angle. This work validates that it is
possible to learn such angles.

sub-folders
------------

- **convergence_analysis**: the code in this package investigates the convergence properties of RDKIT with
  respect to the number of configurations are embedded and then optimized.

- **tests**: Unit tests.
