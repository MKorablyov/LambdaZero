from LambdaZero.environments import MolMDP
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdchem


def gen_parts(nsamples, mol_path):
    molMDP = MolMDP(blocks_file=mol_path)
    atoms = []
    r_factors = []
    print(mol_path)
    num_failed = 0
    for i in range(nsamples):
        molMDP.reset()
        try:
            rs = int(np.random.uniform(4, 20))
            molMDP.random_walk(rs)
            Chem.AddHs(molMDP.molecule.mol)
            num_atoms = rdchem.Mol.GetNumAtoms(molMDP.molecule.mol, onlyExplicit=False)
            r_factor = molMDP.molecule.stem_atmidxs.shape[0]
            atoms.append(int(num_atoms))
            r_factors.append(int(r_factor))
        except Exception as e:
            num_failed += 1

    print(num_failed)
    return atoms, r_factors


vocab_dirs = ["/home/nova/vocabs/pdb_blocks_55.json",
              "/home/nova/vocabs/pdb_blocks_105.json",
              "/home/nova/vocabs/pdb_blocks_210.json",
              "/home/nova/vocabs/pdb_blocks_330.json",
              "/home/nova/vocabs/zinc15_blocks_55.json",
              "/home/nova/vocabs/zinc15_blocks_105.json",
              "/home/nova/vocabs/zinc15_blocks_210.json",
              "/home/nova/vocabs/zinc15_blocks_330.json"]

atoms = []
r_factors = []
num_samples = 2000

for i in range(8):
    if i < 5:
        a, r = gen_parts(int(num_samples/4), vocab_dirs[i])
    else:
        a, r = gen_parts(num_samples, vocab_dirs[i])
    atoms.append(a)
    r_factors.append(r)

graph_titles = ['PDB Blocks BL: 55', 'PDB Blocks BL: 105', 'PDB Blocks BL: 210', 'PDB Blocks BL: 330',
                'Zinc15 Blocks BL: 55', 'Zinc15 Blocks BL: 105', 'Zinc15 Blocks BL: 210', 'Zinc15 Blocks BL: 330']

for i in range(1, 9):
    ax1 = plt.subplot(2, 4, i)
    ax1.set_title(graph_titles[i-1])
    if i < 5:
        ax1.set_xlim([0, 250])
        ax1.set_ylim([0, 100])
    else:
        ax1.set_xlim([0, 120])
        ax1.set_ylim([0, 20])
    plt.xlabel('Number of Atoms')
    plt.ylabel('Branching Factor')
    plt.scatter(atoms[i-1], r_factors[i-1])
    plt.grid()

plt.show()
