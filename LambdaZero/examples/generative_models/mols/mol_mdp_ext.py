from collections import defaultdict
from collections import OrderedDict
import os.path
import numpy as np
import torch
import networkx as nx
import torch_geometric

from LambdaZero.environments.molMDP import BlockMoleculeData, MolMDP
from LambdaZero import chem
from rdkit import Chem
import copy
from collections import Counter

from LambdaZero.examples.generative_models.mols import model_atom, model_block, model_fingerprint


class BlockMoleculeDataExtended(BlockMoleculeData):

    @property
    def mol(self):
        return chem.mol_from_frag(jun_bonds=self.jbonds, frags=self.blocks)[0]

    @property
    def smiles(self):
        return Chem.MolToSmiles(self.mol)

    def copy(self): # shallow copy
        o = BlockMoleculeDataExtended()
        o.blockidxs = list(self.blockidxs)
        o.blocks = list(self.blocks)
        o.slices = list(self.slices)
        o.numblocks = self.numblocks
        o.jbonds = list(self.jbonds)
        o.stems = list(self.stems)
        o.pending_stems = list(self.pending_stems)
        return o

    def as_dict(self):
        return {'blockidxs': self.blockidxs,
                'slices': self.slices,
                'numblocks': self.numblocks,
                'jbonds': self.jbonds,
                'pending_stems': self.pending_stems,
                'stems': self.stems}

    def dump(self) -> dict:
        mol_data = copy.deepcopy({
            "blockidxs": self.blockidxs,
            "slices": self.slices,
            "jbonds": self.jbonds,
            "stems": self.stems,
            "smiles": self.smiles if len(self.blockidxs) > 0 else None,
            "pending_stems": self.pending_stems,
        })
        return mol_data


def node_match(x1, x2):
    return x1["block"] == x2["block"]


def edge_match(x1, x2):
    return x1["bond"] == x2["bond"]


class MolMDPExtended(MolMDP):
    def __init__(self, *args, fix_block_rs: bool = False, **kwargs):
        super(MolMDPExtended, self).__init__(*args, **kwargs)

        # TODO If not fix_block_rs we should actually check some constraints on the blocks
        # E.g. constraints. 1 same smile blocks should have the same stems
        # 2 Each smile block should have a duplicate block starting with each unique stem
        if fix_block_rs:
            """ Duplicate blocks with different set of bonds available. Fix it. """
            smi_bonds = dict()
            for ix, (b_smi, b_rs) in enumerate(zip(self.block_smi, self.block_rs)):
                if b_smi in smi_bonds:
                    smi_bonds[b_smi].update(Counter(b_rs) - smi_bonds[b_smi])
                else:
                    smi_bonds[b_smi] = Counter(b_rs)

            new_blockrs = []
            for ix, (b_smi, b_rs) in enumerate(zip(self.block_smi, self.block_rs)):
                new_rs = b_rs + list((smi_bonds[b_smi] - Counter(b_rs)).elements())
                new_blockrs.append(new_rs)

            self.block_rs = new_blockrs

    def build_translation_table(self, fill_in: bool = True, ):
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
        if fill_in:
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

        # Not necessary if fix kids
        self.translation_same_children = dict()
        for block, block_table in self.translation_table.items():
            crt_block_rs = Counter(self.block_rs[block])
            same = [block]
            for similar_block in np.unique(list(block_table.values())):
                if similar_block != block and Counter(self.block_rs[similar_block]) == crt_block_rs:
                    same.append(similar_block)
            self.translation_same_children[block] = same

    def parents(self, mol=None, keep_possible_children=False):
        """returns all the possible parents of molecule mol (or the current
        molecule if mol is None.
        Returns a list of (BlockMoleculeDataExtended, (block_idx, stem_idx)) pairs such that
        for a pair (m, (b, s)), MolMDPExtended.add_block_to(m, b, s) == mol.
        """
        if len(mol.blockidxs) == 1:
            # If there's just a single block, then the only parent is
            # the empty block with the action that recreates that block
            return [(BlockMoleculeDataExtended(), (mol.blockidxs[0], 0))]

        # Compute the how many blocks each block is connected to
        # Let's keep them ordered so we are sure to add last added block as a remove action
        # Quick fix so we can determine which parent produced the trajectory -
        # This should be the first parent in list
        blocks_degree = OrderedDict()
        for a, b, sa, sb in mol.jbonds[::-1]:
            blocks_degree[b] = 1 if b not in blocks_degree else blocks_degree[b] + 1
            blocks_degree[a] = 1 if a not in blocks_degree else blocks_degree[a] + 1

        # Keep only blocks of degree 1 (those are the ones that could
        # have just been added)
        blocks_degree_1 = [i for i, d in blocks_degree.items() if d == 1]
        # Form new molecules without these blocks
        parent_mols = []
        parent_graphs = []
        parent_molg = self.get_nx_graph(mol, true_block=True)
        parents = defaultdict(list)

        # Remember where we are deleting blocks from
        mol._parent_deleted_block = []
        mol._parent_connected_block = []

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

            mol_connected_block = rbond[0] if rblockidx == rbond[1] else rbond[1]

            # and add it back
            new_mol.stems = [list(i) for i in new_mol.stems] + [stem]

            # Repair blockidx to always have blocks with first stem (from block_rs) connected
            # Need this to compare parents
            connected_bidx = new_mol.blockidxs[stem[0]]
            connected_stem_0 = False

            if len(new_mol.blockidxs) == 1 and stem[1] == self.block_rs[connected_bidx][0]:
                # Could be that its just 1 block and it has duplicate stems
                connected_stem_0 = True
            else:
                for a, b, sa, sb in new_mol.jbonds:
                    if (a == stem[0] and sa == stem[1]) or (b == stem[0] and sb == stem[1]):
                        connected_stem_0 = True
                        break

            if not connected_stem_0:
                prev = connected_bidx
                # Get another stem used by the block
                for a, b, sa, sb in new_mol.jbonds:
                    if a == stem[0]:
                        new_mol.blockidxs[stem[0]] = self.translation_table[connected_bidx][sa]
                        break
                    elif b == stem[0]:
                        new_mol.blockidxs[stem[0]] = self.translation_table[connected_bidx][sb]
                        break

            blockid = mol.blockidxs[rblockidx]

            # Solve duplicates parents same BlockID (check if we already parents for this)
            # Notice difference in graph constructions get_nx_graph with true_block=True
            new_p = True
            new_molg = self.get_nx_graph(new_mol, true_block=True)
            test_blockid = blockid

            if test_blockid in parents:
                for other_p in parents[test_blockid]:
                    if self.graphs_are_isomorphic(other_p, new_molg):
                        new_p = False
                        break

            if not new_p:
                continue
            else:
                parents[test_blockid].append(new_molg)

            # and we have a parent. The stem idx to recreate mol is
            # the last stem, since we appended `stem` in the back of
            # the stem list.
            # We also have to translate the block id to match the bond
            # we broke, see build_translation_table().
            removed_stem_atom = (
                rbond[3] if rblockidx == rbond[1] else rbond[2])

            if removed_stem_atom not in self.translation_table[blockid]:
                raise ValueError('Could not translate removed stem to duplicate or symmetric block.')

            action = (self.translation_table[blockid][removed_stem_atom], len(new_mol.stems) - 1)
            parent_mols.append([new_mol, action])
            mol._parent_deleted_block.append(rblockidx)
            mol._parent_connected_block.append(mol_connected_block)

            # Add other parents because of isomorphism (add to other stem => same block graph)
            if len(new_mol.blockidxs) > 1:
                for test_stem in range(len(new_mol.stems) - 1):
                    test_child = self.add_block_to(new_mol, blockid, test_stem)
                    testg = self.get_nx_graph(test_child, true_block=True)
                    if self.graphs_are_isomorphic(parent_molg, testg):
                        action = (action[0], test_stem)
                        parent_mols.append([new_mol, action])
                        mol._parent_deleted_block.append(rblockidx)
                        mol._parent_connected_block.append(mol_connected_block)

        if not len(parent_mols):
            raise ValueError('Could not find any parents')

        return parent_mols

    def add_block_to(self, mol, block_idx, stem_idx=None, atmidx=None):
        '''out-of-place version of add_block'''
        #assert (block_idx >= 0) and (block_idx <= len(self.block_mols)), "unknown block"
        if mol.numblocks == 0:
            stem_idx = None
        new_mol = mol.copy()
        new_mol.add_block(block_idx,
                          block=self.block_mols[block_idx],
                          block_r=self.block_rs[block_idx],
                          stem_idx=stem_idx, atmidx=atmidx)
        return new_mol

    def remove_jbond_from(self, mol, jbond_idx=None, atmidx=None):
        new_mol = mol.copy()
        new_mol.remove_jbond(jbond_idx, atmidx)
        return new_mol

    def a2mol(self, acts):
        mol = BlockMoleculeDataExtended()
        for i in acts:
          if i[0] >= 0:
            mol = self.add_block_to(mol, *i)
        return mol

    def reset(self):
        self.molecule = BlockMoleculeDataExtended()
        return None


    def post_init(
            self, device, repr_type, include_bonds=False, include_nblocks=False, floatX="float32",
                add_stem_mask=False, donor_features=False, ifcoord=False, one_hot_atom=False, **kwargs
    ):
        self.device = device
        self.repr_type = repr_type
        #self.max_bond_atmidx = max([max(i) for i in self.block_rs])
        self.max_num_atm = max(self.block_natm)
        # see model_block.mol2graph
        self.true_block_set = sorted(set(self.block_smi))
        self.stem_type_offset = np.int32([0] + list(np.cumsum([
            max(self.block_rs[self.block_smi.index(i)])+1 for i in self.true_block_set])))
        self.num_stem_types = self.stem_type_offset[-1]
        self.true_blockidx = [self.true_block_set.index(i) for i in self.block_smi]
        self.num_true_blocks = len(self.true_block_set)
        self.include_nblocks = include_nblocks
        self.include_bonds = include_bonds
        #print(self.max_num_atm, self.num_stem_types)
        self.molcache = {}
        self.floatX = getattr(torch, floatX)
        self.add_stem_mask = add_stem_mask
        self.donor_features = donor_features
        self.ifcoord = ifcoord
        self.one_hot_atom = one_hot_atom

    def mols2batch(self, mols):
        if self.repr_type == 'block_graph':
            return model_block.mols2batch(mols, self)
        elif self.repr_type == 'atom_graph':
            return model_atom.mols2batch(mols, self)
        elif self.repr_type == 'morgan_fingerprint':
            return model_fingerprint.mols2batch(mols, self)

    def mol2repr(self, mol=None):
        if mol is None:
            mol = self.molecule

        if self.repr_type == 'block_graph':
            r = model_block.mol2graph(mol, self, self.floatX,
                                      bonds=self.include_bonds,
                                      nblocks=self.include_nblocks)
        elif self.repr_type == 'atom_graph':
            r = model_atom.mol2graph(mol, self, self.floatX,
                                     bonds=self.include_bonds,
                                     nblocks=self.include_nblocks,
                                     add_stem_mask=self.add_stem_mask,
                                     donor_features=self.donor_features,
                                     ifcoord=self.ifcoord,
                                     one_hot_atom=self.one_hot_atom)
        elif self.repr_type == 'morgan_fingerprint':
            r = model_fingerprint.mol2fp(mol, self, self.floatX)

        return r

    def load(self, state: dict) -> BlockMoleculeDataExtended:
        self.reset()
        state = copy.deepcopy(state)
        self.molecule.blockidxs = state["blockidxs"]  # indexes of every block
        self.molecule.slices = state["slices"]  # atom index at which every block starts
        self.molecule.jbonds = state["jbonds"]  # [block1, block2, bond1, bond2]
        self.molecule.stems = state["stems"]  # [block1, bond1]
        self.molecule.pending_stems = state["pending_stems"]  # [block1, bond1]
        self.molecule.numblocks = len(self.molecule.blockidxs)
        self.molecule.blocks = [self.block_mols[idx] for idx in self.molecule.blockidxs]  # rdkit
        return self.molecule

    def unique_id(self, mol: BlockMoleculeData) -> Counter:
        bonds = []
        block_smi = self.block_smi
        if len(mol.jbonds) > 0:
            for jbond in mol.jbonds:
                blockidx0, blockidx1 = mol.blockidxs[jbond[0]], mol.blockidxs[jbond[1]]
                bonds.append((block_smi[blockidx0], jbond[2], jbond[3], block_smi[blockidx1]))
                bonds.append((block_smi[blockidx1], jbond[3], jbond[2], block_smi[blockidx0]))
        else:
            bonds.append((block_smi[mol.blockidxs[0]]))

        mol_data = Counter(bonds)
        return mol_data

    def num_act_stems(self, mol: BlockMoleculeDataExtended):
        if len(mol.blockidxs) <= 1:
            return 1
        elif len(mol.stems) == 0:
            return 0
        else:
            return len(mol.stems)

    def get_nx_graph(self, mol: BlockMoleculeData, true_block=False):
        true_blockidx = self.true_blockidx

        G = nx.DiGraph()
        blockidxs = [true_blockidx[xx] for xx in mol.blockidxs] if true_block else mol.blockidxs

        G.add_nodes_from([(ix, {"block": blockidxs[ix]}) for ix in range(len(blockidxs))])

        if len(mol.jbonds) > 0:
            edges = []
            for jbond in mol.jbonds:
                edges.append((jbond[0], jbond[1],
                              {"bond": [jbond[2], jbond[3]]}))
                edges.append((jbond[1], jbond[0],
                              {"bond": [jbond[3], jbond[2]]}))
            G.add_edges_from(edges)
        return G

    def graphs_are_isomorphic(self, g1, g2):
        return nx.algorithms.is_isomorphic(g1, g2, node_match=node_match, edge_match=edge_match)


def test_mdp_parent():
    datasets_dir, programs_dir, summaries_dir = get_external_dirs()
    bpath = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    mdp = MolMDPExtended(bpath)
    mdp.build_translation_table()
    import tqdm
    rng = np.random.RandomState(142)
    nblocks = mdp.num_blocks

    # First let's test that the parent-finding method is
    # correct. i.e. let's test that the (mol, (parent, action)) pairs
    # are such that add_block_to(parent, action) == mol
    for i in tqdm.tqdm(range(10000)):
        mdp.molecule = mol = BlockMoleculeDataExtended()
        nblocks = rng.randint(1, 10)
        for i in range(nblocks):
            if len(mol.blocks) and not len(mol.stems): break
            mdp.add_block(rng.randint(nblocks), rng.randint(max(1, len(mol.stems))))
        parents = mdp.parents(mol)
        s = mol.smiles
        for p, (a,b) in parents:
            c = mdp.add_block_to(p, a, b)
            if c.smiles != s:
                # SMILES might differ but this might still be the same mol
                # we can check this way but its a bit more costly
                assert c.mol.HasSubstructMatch(mol.mol)

    # Now let's test whether we can always backtrack to the root from
    # any molecule without any errors
    for i in tqdm.tqdm(range(10000)):
        mdp.molecule = mol = BlockMoleculeDataExtended()
        nblocks = rng.randint(1, 10)
        for i in range(nblocks):
            if len(mol.blocks) and not len(mol.stems): break
            mdp.add_block(rng.randint(nblocks), rng.randint(max(1, len(mol.stems))))
        while len(mol.blocks):
            parents = mdp.parents(mol)
            mol = parents[rng.randint(len(parents))][0]

if __name__ == '__main__':
    from LambdaZero.utils import get_external_dirs
    test_mdp_parent()
