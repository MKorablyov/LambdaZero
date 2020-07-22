"""
The goal of this script is to make sure that the embedding energy remains the same when we add/remove/add-again
hydrogen atoms and constrain embed. The code illustrates issues with ConstrainEmbed as well as the proposed fix.
Executing the script prints out intermediate results to help understand the story.
"""
import os

import numpy as np
from rdkit.Chem.AllChem import ConstrainedEmbed, MMFFOptimizeMolecule, MolToSmiles
from rdkit.Chem.rdchem import Mol

import LambdaZero.utils
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_mmff_force_field,
    get_mmff_energy,
    get_lowest_energy_and_mol_with_hydrogen,
    get_atomic_symbols,
)

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

number_of_blocks = 5
num_conf = 100
random_seed = 1231

if __name__ == "__main__":
    print("#=============================================================")
    print("# This script illustrates issues with constrained optimization")
    print("#=============================================================")

    np.random.seed(random_seed)
    molMDP = MolMDP(blocks_file=blocks_file)
    molMDP.reset()
    molMDP.random_walk(number_of_blocks)

    mol = molMDP.molecule.mol
    symbols = get_atomic_symbols(mol)

    print(f" - A random molecule has been generated, smiles = {MolToSmiles(mol)}")

    min_energy, mol_with_hydrogens, _ = get_lowest_energy_and_mol_with_hydrogen(
        mol, num_conf, random_seed=random_seed
    )

    conf_id = mol_with_hydrogens.GetConformer().GetId()

    print(f" - This molecule is embedded and relaxed. The energy of the best conformer is {min_energy:5.3f} kcal/mol")

    print(f" - We run various checks: ")
    energy_from_force_field = get_mmff_energy(mol_with_hydrogens, conf_id=conf_id)
    print(f"      o energy from force field: {energy_from_force_field:5.3f} kcal/mol  (error: {min_energy-energy_from_force_field:5.3e})")

    #  If I relax again, does it move away?

    new_mol_with_hydrogens = Mol(mol_with_hydrogens)
    converged = MMFFOptimizeMolecule(new_mol_with_hydrogens, mmffVariant="MMFF94", maxIters=200)
    energy_after_more_relax = get_mmff_energy(new_mol_with_hydrogens, conf_id=conf_id)
    print(f"      o energy after new relax: {energy_after_more_relax:5.3f} kcal/mol   (error: {min_energy-energy_after_more_relax:5.3e})")

    # Does ConstrainEmbed break everything?

    mol_with_hydrogens_for_contrained_embed = Mol(mol_with_hydrogens)
    ConstrainedEmbed(
        mol_with_hydrogens_for_contrained_embed,
        mol_with_hydrogens,
        useTethers=True,
        randomseed=random_seed,
        getForceField=get_mmff_force_field,
    )

    constrained_embed_conf_id = mol_with_hydrogens_for_contrained_embed.GetConformer().GetId()
    energy_after_constrained_embed = get_mmff_energy(mol_with_hydrogens_for_contrained_embed, conf_id=constrained_embed_conf_id)
    de = min_energy-energy_after_constrained_embed
    p1 = mol_with_hydrogens.GetConformer().GetPositions()
    p2 = mol_with_hydrogens_for_contrained_embed.GetConformer().GetPositions()

    print("  - We use ConstrainedEmbed to constrain ALL ATOMS")
    print(f"       o energy after ConstrainEmbed: {energy_after_constrained_embed:5.3f} kcal/mol")
    print(f"          ---> difference: {de:5.3e} kcal/mol   THIS IS BIG!")
    print(f"       o norm position diff: {np.linalg.norm(p1-p2):5.3e} [distance unit]")
    print(f" ---> ConstrainEmbed hard codes constraints that are too weak, leading to large energy error")


    print(f" ")
    print(f" Let's build our own constraints!")
    print(f" ")

    print(f"  Force parameter     |Energy Error| (kcal/mol)     position error")
    print(f"==================================================================")

    list_force_constants = 10 ** np.array(np.arange(7))

    for force_constant in list_force_constants:

        new_mol = Mol(mol_with_hydrogens)
        p_before = new_mol.GetConformer().GetPositions()
        force_field = get_mmff_force_field(new_mol)

        for idx in range(new_mol.GetNumAtoms()):
            force_field.MMFFAddPositionConstraint(idx, maxDispl=1e-6, forceConstant=int(force_constant))

        force_field.Initialize()
        force_field.Minimize(maxIts=200, forceTol=1e-6, energyTol=1e-08)
        p_after = new_mol.GetConformer().GetPositions()

        relaxed_energy = get_mmff_energy(new_mol, conf_id=conf_id)
        energy_error = np.abs(relaxed_energy - min_energy)
        position_error = np.linalg.norm(p_before-p_after)

        print(f"       {force_constant:3.1e}                {energy_error:5.3e}              {position_error:5.3e}")

    print(f"----> The Force parameter has to be large to ensure good convergence (ConstrainEmbed sets it to 100)")
