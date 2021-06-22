from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
from rdkit import Chem
import copy

from LambdaZero import chem


class BlockMoleculeData:
    # todo: make properties fast
    def __init__(self):
        self.blockidxs = []       # indexes of every block
        self.blocks = []          # rdkit molecule objects for every
        self.slices = [0]         # atom index at which every block starts
        self.numblocks = 0
        self.jbonds = []          # [block1, block2, bond1, bond2]
        self.stems = []           # [block1, bond1]
        self._mol = None

    def add_block(self, block_idx, block, block_r, stem_idx, atmidx):
        """
        :param block_idx:
        :param block:
        :param block_r:
        :param stem_idx:
        :param atmidx:
        :return:
        """
        self.blockidxs.append(block_idx)
        self.blocks.append(block)
        self.slices.append(self.slices[-1] + block.GetNumAtoms())
        self.numblocks += 1
        [self.stems.append([self.numblocks-1,r]) for r in block_r[1:]]

        if len(self.blocks)==1:
            self.stems.append([self.numblocks-1, block_r[0]])
        else:
            if stem_idx is None:
                assert atmidx is not None, "need stem or atom idx"
                stem_idx = np.where(self.stem_atmidxs==atmidx)[0][0]
            else:
                assert atmidx is None, "can't use stem and atom indices at the same time"

            stem = self.stems[stem_idx]
            bond = [stem[0], self.numblocks-1, stem[1], block_r[0]]
            self.stems.pop(stem_idx)
            self.jbonds.append(bond)
            # destroy properties
            self._mol = None
        return None

    def delete_blocks(self, block_mask):
        """
        :param block_mask:
        :return:
        """

        # update number of blocks
        self.numblocks = np.sum(np.asarray(block_mask, dtype=np.int32))
        self.blocks = list(np.asarray(self.blocks)[block_mask])
        self.blockidxs = list(np.asarray(self.blockidxs)[block_mask])

        # update junction bonds
        reindex = np.cumsum(np.asarray(block_mask,np.int32)) - 1
        jbonds = []
        for bond in self.jbonds:
            if block_mask[bond[0]] and block_mask[bond[1]]:
                jbonds.append(np.array([reindex[bond[0]], reindex[bond[1]], bond[2], bond[3]]))
        self.jbonds = jbonds

        # update r-groups
        stems = []
        for stem in self.stems:
            if block_mask[stem[0]]:
                stems.append(np.array([reindex[stem[0]],stem[1]]))
        self.stems = stems

        # update slices
        natms = [block.GetNumAtoms() for block in self.blocks]
        self.slices = [0] + list(np.cumsum(natms))

        # destroy properties
        self._mol = None
        return reindex

    def remove_jbond(self, jbond_idx=None, atmidx=None):

        if jbond_idx is None:
            assert atmidx is not None, "need jbond or atom idx"
            jbond_idx = np.where(self.jbond_atmidxs == atmidx)[0][0]
        else:
            assert atmidx is None, "can't use stem and atom indices at the same time"

        # find index of the junction bond to remove
        jbond = self.jbonds.pop(jbond_idx)

        # find the largest connected component; delete rest
        jbonds = np.asarray(self.jbonds, dtype=np.int32)
        jbonds = jbonds.reshape([len(self.jbonds),4]) # handle the case when single last jbond was deleted
        graph = csr_matrix((np.ones(self.numblocks-2),
                            (jbonds[:,0], jbonds[:,1])),
                           shape=(self.numblocks, self.numblocks))
        _, components = connected_components(csgraph=graph, directed=False, return_labels=True)
        block_mask = components==np.argmax(np.bincount(components))
        reindex = self.delete_blocks(block_mask)

        if block_mask[jbond[0]]:
            stem = np.asarray([reindex[jbond[0]], jbond[2]])
        else:
            stem = np.asarray([reindex[jbond[1]], jbond[3]])
        self.stems.append(stem)
        atmidx = self.slices[stem[0]] + stem[1]
        return atmidx

    @property
    def stem_atmidxs(self):
        stems = np.asarray(self.stems)
        if stems.shape[0]==0:
            stem_atmidxs = np.array([])
        else:
            stem_atmidxs = np.asarray(self.slices)[stems[:,0]] + stems[:,1]
        return stem_atmidxs

    @property
    def jbond_atmidxs(self):
        jbonds = np.asarray(self.jbonds)
        if jbonds.shape[0]==0:
            jbond_atmidxs = np.array([])
        else:
            jbond_atmidxs = np.stack([np.concatenate([np.asarray(self.slices)[jbonds[:,0]] + jbonds[:,2]]),
                                      np.concatenate([np.asarray(self.slices)[jbonds[:,1]] + jbonds[:,3]])],1)
        return jbond_atmidxs

    @property
    def mol(self):
        if self._mol == None:
            self._mol, _ = chem.mol_from_frag(jun_bonds=self.jbonds, frags=self.blocks)
        return self._mol

    @property
    def smiles(self):
        return Chem.MolToSmiles(self.mol)

    def copy(self):
        """shallow copy, sufficient for most purposes"""
        o = BlockMoleculeData()
        o.blockidxs = list(self.blockidxs)
        o.blocks = list(self.blocks)
        o.slices = list(self.slices)
        o.numblocks = self.numblocks
        o.jbonds = list(self.jbonds)
        o.stems = list(self.stems)
        return o

class MolMDP:
    def __init__(self, blocks_file):
        blocks = pd.read_json(blocks_file)
        self.block_smi = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()
        self.block_nrs = np.asarray([len(r) for r in self.block_rs])
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in blocks["block_smi"]]
        self.block_natm = np.asarray([b.GetNumAtoms() for b in self.block_mols])
        #self.num_blocks = len(self.block_smi)
        self.reset()

    @property
    def num_blocks(self):
        "number of possible building blocks in molMDP"
        return len(self.block_smi)

    def reset(self):
        self.molecule = BlockMoleculeData()
        return None

    def add_block(self, block_idx, stem_idx=None, atmidx=None):
        assert (block_idx >= 0) and (block_idx <= len(self.block_mols)), "unknown block"
        self.molecule.add_block(block_idx,
                                block=self.block_mols[block_idx],
                                block_r=self.block_rs[block_idx],
                                stem_idx=stem_idx, atmidx=atmidx)
        #assert self.molecule.mol is not None, "returning not a molecule"
        return None

    def remove_jbond(self, jbond_idx=None, atmidx=None):
        atmidx = self.molecule.remove_jbond(jbond_idx, atmidx)
        #assert self.molecule.mol is not None, "returning not a molecule"
        return atmidx

    def random_walk(self, length):
        done = False
        while not done:
            if self.molecule.numblocks==0:
                block_idx = np.random.choice(np.arange(self.num_blocks))
                stem_idx = None
                self.add_block(block_idx=block_idx, stem_idx=stem_idx)
                if self.molecule.numblocks >= length:
                    if self.molecule.slices[-1] > 1:
                        done = True
                    else:
                        self.reset()
            elif len(self.molecule.stems) > 0:
                block_idx = np.random.choice(np.arange(self.num_blocks))
                stem_idx = np.random.choice(len(self.molecule.stems))
                self.add_block(block_idx=block_idx, stem_idx=stem_idx)
                if self.molecule.numblocks >= length: done = True
            else:
                self.reset()
        #assert self.molecule.mol is not None, "returning not a molecule"

    def parents(self, mol=None):
        """returns all the possible parents of molecule mol (or the current
        molecule if mol is None.
        Returns a list of (BlockMoleculeData, (block_idx, stem_idx)) pairs such that
        for a pair (m, (b, s)), MolMDP.add_block_to(m, b, s) == mol.
        """
        if len(mol.blockidxs) == 1:
            # If there's just a single block, then the only parent is
            # the empty block with the action that recreates that block
            return [(BlockMoleculeData(), (mol.blockidxs[0], 0))]

        # Compute the how many blocks each block is connected to
        blocks_degree = defaultdict(int)
        for a,b,_,_ in mol.jbonds:
            blocks_degree[a] += 1
            blocks_degree[b] += 1
        # Keep only blocks of degree 1 (those are the ones that could
        # have just been added)
        blocks_degree_1 = [i for i, d in blocks_degree.items() if d == 1]
        # Form new molecules without these blocks
        parent_mols = []

        for rblockidx in blocks_degree_1:
            new_mol = mol.copy()
            # find which bond we're removing
            removed_bonds = [(jbidx, bond) for jbidx, bond in enumerate(new_mol.jbonds)
                             if rblockidx in bond[:2]]
            assert len(removed_bonds) == 1
            rjbidx, rbond = removed_bonds[0]
            # Pop the bond
            new_mol.jbonds.pop(rjbidx)
            # Remove the block
            mask = np.ones(len(new_mol.blockidxs), dtype=np.bool)
            mask[rblockidx] = 0
            reindex = new_mol.delete_blocks(mask)
            # reindex maps old blockidx to new blockidx, since the
            # block the removed block was attached to might have its
            # index shifted by 1.

            # Compute which stem the bond was using
            stem = ([reindex[rbond[0]], rbond[2]] if rblockidx == rbond[1] else
                    [reindex[rbond[1]], rbond[3]])
            # and add it back
            new_mol.stems = [list(i) for i in new_mol.stems] + [stem]
            # and we have a parent.

            # Now we compute the action from parent -> mol. This is
            # the atom the bond was attached to:
            removed_stem_atom = (
                rbond[3] if rblockidx == rbond[1] else rbond[2])
            # the removed block's id
            blockid = mol.blockidxs[rblockidx]
            if removed_stem_atom not in self.translation_table[blockid]:
                raise ValueError('Could not translate removed stem to duplicate or symmetric block.')
            # action = (block_idx, stem_idx)

            # TODO This should be fixed with the bug fix in the env
            if self.translation_table[blockid][removed_stem_atom] != blockid:
                # QUICK FIX: Skip if not the same blockidx (so the mol can be reconsctructed)
                continue

            action = (
                # We have to translate the block id to match the bond
                # we broke, see build_translation_table().
                self.translation_table[blockid][removed_stem_atom],
                # The stem idx to recreate mol is the last stem,
                # since we appended `stem` in the back of the
                # stem list.
                len(new_mol.stems) - 1)
            parent_mols.append([new_mol, action])
        if not len(parent_mols) and len(mol.blocks):
            # The molecule was not the empty molecule and we didn't
            # find any parents? This should not happen.
            raise ValueError('Could not find any parents')
        return parent_mols

    def build_translation_table(self):
        """build a symmetry mapping for blocks. Necessary to compute parent transitions"""
        self.translation_table = {}
        for blockidx in range(len(self.block_mols)):
            # Blocks have multiple ways of being attached. By default,
            # a new block is attached to the target stem by attaching
            # it's kth atom, where k = block_rs[new_block_idx][0].
            # When computing a reverse action (from a parent), we may
            # wish to attach the new block to a different atom. In
            # the blocks library, there are duplicates of the same
            # block but with block_rs[block][0] set to a different
            # atom. Thus, for the reverse action we have to find out
            # which duplicate this corresponds to.

            # Here, we compute, for block blockidx, what is the index
            # of the duplicate block, if someone wants to attach to
            # atom x of the block.
            # So atom_map[x] == bidx, such that block_rs[bidx][0] == x
            atom_map = {}
            for j in range(len(self.block_mols)):
                if self.block_smi[blockidx] == self.block_smi[j]:
                    atom_map[self.block_rs[j][0]] = j
            self.translation_table[blockidx] = atom_map

        # We're still missing some "duplicates", as some might be
        # symmetric versions of each other. For example, block CC with
        # block_rs == [0,1] has no duplicate, because the duplicate
        # with block_rs [1,0] would be a symmetric version (both C
        # atoms are the "same").

        # To test this, let's create nonsense molecules by attaching
        # duplicate blocks to a Gold atom, and testing whether they
        # are the same.
        gold = Chem.MolFromSmiles('[Au]')
        # If we find that two molecules are the same when attaching
        # them with two different atoms, then that means the atom
        # numbers are symmetries. We can add those to the table.
        for blockidx in range(len(self.block_mols)):
            for j in self.block_rs[blockidx]:
                if j not in self.translation_table[blockidx]:
                    symmetric_duplicate = None
                    for atom, block_duplicate in self.translation_table[blockidx].items():
                        molA, _ = chem.mol_from_frag(
                            jun_bonds=[[0,1,0,j]],
                            frags=[gold, self.block_mols[blockidx]])
                        molB, _ = chem.mol_from_frag(
                            jun_bonds=[[0,1,0,atom]],
                            frags=[gold, self.block_mols[blockidx]])
                        if (Chem.MolToSmiles(molA) == Chem.MolToSmiles(molB) or
                            molA.HasSubstructMatch(molB)):
                            symmetric_duplicate = block_duplicate
                            break
                    if symmetric_duplicate is None:
                        raise ValueError('block', blockidx, self.block_smi[blockidx],
                                         'has no duplicate for atom', j,
                                         'in position 0, and no symmetrical correspondance')
                    self.translation_table[blockidx][j] = symmetric_duplicate

    def load(self, state: dict) -> BlockMoleculeData:
        self.reset()
        self.molecule.blockidxs = state["blockidxs"]  # indexes of every block
        self.molecule.slices = state["slices"]  # atom index at which every block starts
        self.molecule.jbonds = state["jbonds"]  # [block1, block2, bond1, bond2]
        self.molecule.stems = state["stems"]  # [block1, bond1]
        self.molecule.numblocks = len(self.molecule.blockidxs)
        self.molecule.blocks = [self.block_mols[idx] for idx in self.molecule.blockidxs]  # rdkit
        return self.molecule

    def dump(self) -> dict:
        mol_data = copy.deepcopy({
            "blockidxs": self.molecule.blockidxs,
            "slices": self.molecule.slices,
            "jbonds": self.molecule.jbonds,
            "stems": self.molecule.stems,
            "smiles": self.molecule.smiles
        })
        return mol_data