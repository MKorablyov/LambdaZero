"""
This script shows that the num_conf embedding energies are in a consistent order. That is to say,
for num_conf1 < num_conf2, all the energies produced with num_conf1 are in the same order as the
first num_conf1 energies calculated with num_conf2.
"""
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import SanitizeMol
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.environments.molMDP import MolMDP

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

# computation parameters
number_of_blocks = 5
random_seed = 123
small_num_conf = 5
large_num_conf = 10
max_iters = 200

if __name__ == "__main__":

    np.random.seed(random_seed)
    reference_molMDP = MolMDP(blocks_file=blocks_file)

    # Confirm that the the embeddings are in the same order irrespective of num_conf
    number_of_successes = 0
    counter = 0
    while number_of_successes == 0:
        counter += 1
        print(f" - Attempt {counter}")
        reference_molMDP.reset()
        reference_molMDP.random_walk(number_of_blocks)
        augmented_mol = Chem.AddHs(reference_molMDP.molecule.mol)
        SanitizeMol(augmented_mol)
        AllChem.EmbedMultipleConfs(
            augmented_mol, numConfs=small_num_conf, randomSeed=random_seed
        )
        list_tuples_energy_converged = AllChem.MMFFOptimizeMoleculeConfs(
            augmented_mol, mmffVariant="MMFF94", maxIters=max_iters
        )
        number_of_successes = np.sum([t[0] == 0 for t in list_tuples_energy_converged])

    reference_mol = reference_molMDP.molecule.mol

    list_convergence = []
    for num_conf in tqdm([small_num_conf, large_num_conf]):
        augmented_mol = Chem.AddHs(reference_mol)
        SanitizeMol(augmented_mol)

        AllChem.EmbedMultipleConfs(
            augmented_mol, numConfs=num_conf, randomSeed=random_seed
        )
        list_tuples_energy_converged = AllChem.MMFFOptimizeMoleculeConfs(
            augmented_mol, mmffVariant="MMFF94", maxIters=max_iters
        )

        list_convergence.append(list_tuples_energy_converged)

    for energy_tuples1, energy_tuples2 in zip(
        list_convergence[:-1], list_convergence[1:]
    ):
        sub_energy_tuples2 = energy_tuples2[: len(energy_tuples1)]

        for (c1, e1), (c2, e2) in zip(energy_tuples1, sub_energy_tuples2):
            assert c1 == c2, "convergence behavior is not the same"
            assert e1 == e2, "energy is not the same"

    print(" --> energies produced with num_conf are in consistent order")
