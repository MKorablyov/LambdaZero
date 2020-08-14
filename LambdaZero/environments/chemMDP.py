import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from copy import copy


Aromatic_Rings = ['C1C=CC=C1', 'c1ccccc1'] #C1 in pentadiene corresponds to the sp3 carbon
Aliphatic_Rings = ['C1CC1', 'C1CCC1', 'C1CCCC1', 'C1CCCCC1', 'C1CCCCCC1', 'C1CCCCCCC1', 'C1CCCCCCCC1']
Atoms = ['6', '7', '8', '9', '15', '16', '17'] #, '35', '53'] # C, N, O, F, P, S, Cl, Br, I

# fixme temporary fix for valence not yet here - some chemistry issues

class chemMDP:
    def __init__(self):
        self.error = 0
        self.success = 0
        self.actions = [self.add_atom, self.edit_atom, self.delete_atom, self.edit_bond, self.delete_bond, self.add_ring]
        self.reset()

    def reset(self):
        self.molecule = Chem.RWMol()
        return None

    def add_atom(self, atom_indx=None, new_atom=6): #default is carbon
        # adds an atom to atom_indx[0] with single bond
        if self.molecule.GetNumAtoms() == 0:
            self.molecule.AddAtom(Chem.Atom(new_atom))
        else:
            atom_indx_new = self.molecule.AddAtom(Chem.Atom(new_atom))
            self.molecule.AddBond(atom_indx[0], atom_indx_new, Chem.BondType.SINGLE)
        Chem.SanitizeMol(self.molecule)
        return None

    def edit_atom(self, atom_indx, new_atom=7):
        # mutates atom of atom_indx[0] to new_atom type
        if self.molecule.GetAtomWithIdx(atom_indx[0]).GetAtomicNum() == new_atom: return None

        atom_indx_new = self.molecule.AddAtom(Chem.Atom(new_atom))
        if self.molecule.GetNumAtoms() > 2:
            for neighbor in self.molecule.GetAtomWithIdx(atom_indx[0]).GetNeighbors():
                neighbor_indx = neighbor.GetIdx()
                self.molecule.AddBond(atom_indx_new, neighbor_indx, self.molecule.GetBondBetweenAtoms(neighbor_indx, atom_indx[0]).GetBondType())
        elif self.molecule.GetNumAtoms() == 2:
            # i.e. started with just 1 atom
            self.molecule.RemoveAtom(atom_indx[0])
        # or if it was an empty molecule, now we added an atom, so do nothing
        # self.molecule.ReplaceAtom(atom_indx[0], Chem.Atom(new_atom), updateLabel=True, preserveProps=True) # doesn't work
        Chem.SanitizeMol(self.molecule)
        return None

    def delete_atom(self, atom_indx, *args):
        # if self.molecule.GetNumAtoms() == 0: self.reset()
        #if self.molecule.GetAtomWithIdx(atom_indx[0]).IsInRing(): raise Exception('error, cannot delete atoms in ring')
        self.molecule.RemoveAtom(atom_indx[0])
        if self.molecule.GetNumAtoms() == 0: self.reset()
        else:
            # removing the bond often generates fragments, remove the smallest fragment(s)
            fragments = list(Chem.GetMolFrags(self.molecule, asMols=True))
            fragments.sort(reverse=True, key=lambda frag: frag.GetNumAtoms())
            self.molecule = Chem.RWMol(fragments[0])
            Chem.SanitizeMol(self.molecule)
        return None

    def edit_bond(self, atom_indx, new_bond_type):
        # changes bond type from single to double to triple
        # atom_indx is a list of 2 atoms of beginning and end of the bond
        # altnernatively if we use bond_indx: GetBeginAtomIdx(bond_indx)
        if self.molecule.GetNumAtoms() <= 1:
            self.reset()
            return None #raise Exception('Cant edit bond when there is none')
        if int(self.molecule.GetBondBetweenAtoms(atom_indx[0], atom_indx[1]).GetBondTypeAsDouble()) == new_bond_type:
            return None
        self.molecule.RemoveBond(atom_indx[0], atom_indx[1])

        if new_bond_type == 1:
            self.molecule.AddBond(atom_indx[0], atom_indx[1], Chem.BondType.SINGLE)
        #elif new_bond_type == 2:
        #    self.molecule.AddBond(atom_indx[0], atom_indx[1], Chem.BondType.AROMATIC)
        elif new_bond_type == 2:

            atom_1 = self.molecule.GetAtomWithIdx(atom_indx[0])
            atom_2 = self.molecule.GetAtomWithIdx(atom_indx[1])

            # no allenes at all
            for neighbor in atom_1.GetNeighbors():
                if self.molecule.GetBondBetweenAtoms(neighbor.GetIdx(), atom_indx[0]).GetBondTypeAsDouble() == 2.0 or 1.5:
                        raise Exception('Cant connect two double bonds')
                for neighbor in atom_2.GetNeighbors():
                    if self.molecule.GetBondBetweenAtoms(neighbor.GetIdx(), atom_indx[1]).GetBondTypeAsDouble() == 2.0 or 1.5:
                        raise Exception('Cant connect two double bonds')

            # no double-double bond in a ring, rdkit doesn't check for this - but this does NOT prevent 2-double bonds in a ring
            # if (atom_1.IsInRing() and atom_2.IsInRing()): # or rules out feasible molecules
            #     if atom_1.GetIsAromatic() or atom_2.GetIsAromatic():
            #         raise Exception('Cant connect two double bonds')
            #     for neighbor in atom_1.GetNeighbors():
            #         if self.molecule.GetBondBetweenAtoms(neighbor.GetIdx(), atom_indx[0]).GetBondTypeAsDouble() == 2.0 or 1.5:
            #             raise Exception('Cant connect two double bonds')
            #     for neighbor in atom_2.GetNeighbors():
            #         if self.molecule.GetBondBetweenAtoms(neighbor.GetIdx(), atom_indx[1]).GetBondTypeAsDouble() == 2.0 or 1.5:
            #             raise Exception('Cant connect two double bonds')

            self.molecule.AddBond(atom_indx[0], atom_indx[1], Chem.BondType.DOUBLE)
        elif new_bond_type == 3:
            self.molecule.AddBond(atom_indx[0], atom_indx[1], Chem.BondType.TRIPLE)
        else:
            raise Exception('error, needs to choose a bond')
        Chem.SanitizeMol(self.molecule)
        return None

    def delete_bond(self, atom_indx):
        # find index of the junction bond to remove
        # atom_indx is a list of 2 atoms of beginning and end of the bond
        # altnernatively if we use bond_indx: bond = GetBondWithIndx(bond_indx), bond.GetBeginAtomIdx(bond_indx)
        # do not delete ring atoms
        if self.molecule.GetNumAtoms() <= 1 or self.molecule.GetNumBonds() == 0:
            self.reset()
            return None #raise Exception('Cant delete bond when there is none')
        #if self.molecule.GetBondBetweenAtoms(atom_indx[0], atom_indx[1]).IsInRing(): raise Exception('error, cannot remove bonds in ring')
        self.molecule.RemoveBond(atom_indx[0], atom_indx[1])
        #removing the bond generates fragments, remove the smallest fragment(s), if any
        fragments = list(Chem.GetMolFrags(self.molecule, asMols=True))
        fragments.sort(reverse=True, key=lambda frag: frag.GetNumAtoms())
        self.molecule = Chem.RWMol(fragments[0])
        Chem.SanitizeMol(self.molecule)

        return None

    def add_ring(self, atom_indx=None, ring_size=6, aromatic=True, ring_indx=1):
        # atom_indx is a list of atom indices to add the ring
        # ring_indx should be where to add the ring - but for pentadiene only; auto is alpha position (1)
        # adds aliphatic or aromatic ring of size 3-7, if location is on a bond, add fused ring

        if aromatic and (ring_size <= 4 or ring_size >= 7): # currently only cyclopentadiene and benzene
            raise Exception('error, smallest aromatic ring is pentadiene and largest aromatic ring is benzene')
        if ring_size > 9 or ring_size < 3:
            raise Exception('error, ring too large or too small')

        if aromatic:
            ring = Chem.MolFromSmiles(Aromatic_Rings[ring_size-5]) # currently only cyclopentadiene and benzene
            ring = Chem.RWMol(ring)
        else:
            ring = Chem.MolFromSmiles(Aliphatic_Rings[ring_size-3]) # cyclopropane, cyclobutane, etc.
            ring = Chem.RWMol(ring)

        if self.molecule.GetNumAtoms() == 0:
            # if it's initlization as a ring
            self.molecule = ring
        elif not aromatic:
            # for aliphatic rings, remove any atoms (len(atom_indx) times) and stitch the molcules; keep the bond structure at the original bond to add on
            if len(atom_indx) == ring_size:
                # ring closure of the ends
                if bool(self.molecule.GetBondBetweenAtoms(atom_indx[0], atom_indx[-1])):
                    self.molecule.AddBond(atom_indx[0], atom_indx[-1], Chem.BondType.SINGLE)
            else:
                # normal fusing rings
                for i in range(len(atom_indx)):
                    ring.RemoveAtom(0)
                # stitch molecule and ring together
                num_atom = self.molecule.GetNumAtoms()
                self.molecule = Chem.RWMol(Chem.CombineMols(self.molecule, ring))
                # make new bonds
                self.molecule.AddBond(atom_indx[0], num_atom, Chem.BondType.SINGLE)
                self.molecule.AddBond(atom_indx[-1], num_atom + ring_size - (1 + len(atom_indx)), Chem.BondType.SINGLE)
        elif ring_size == 5:
            # for cyclopentadiene; ring_indx refers to position 1, 2, 3 in cyclopentadiene. note adding onto C3 is different from C2
            num_atom = self.molecule.GetNumAtoms()
            if len(atom_indx) == 1:
                # add ring onto 1 atom
                ring.RemoveAtom(ring_indx)
                # stitch molecule and ring together
                self.molecule = Chem.RWMol(Chem.CombineMols(self.molecule, ring))
                if ring_indx == 0:
                    # add onto sp3 carbon by making new bonds to the original molecule
                    self.molecule.AddBond(atom_indx[0], num_atom, Chem.BondType.SINGLE)
                    self.molecule.AddBond(atom_indx[0], num_atom + ring_size - (1 + len(atom_indx)),
                                          Chem.BondType.SINGLE) # for cosistency one can employ atom_indx[-1], but there's only 1 element
                elif ring_indx == 1:
                    # add onto alpha position
                    self.molecule.AddBond(atom_indx[0], num_atom, Chem.BondType.SINGLE)
                    self.molecule.AddBond(atom_indx[0], num_atom + 1, Chem.BondType.DOUBLE)
                else:
                    # to beta position
                    self.molecule.AddBond(atom_indx[0], num_atom + 1, Chem.BondType.DOUBLE)
                    self.molecule.AddBond(atom_indx[0], num_atom + 2, Chem.BondType.SINGLE)
            elif len(atom_indx) == 2:
                # add ring onto a bond (2 atoms)
                # remove the two atoms in cyclopentadiene
                ring.RemoveAtom(ring_indx)
                ring.RemoveAtom(ring_indx)
                # stitch molecule and ring together
                self.molecule = Chem.RWMol(Chem.CombineMols(self.molecule, ring))
                # two cases: 1) bond to add on is aliphatic, or 2) bond to add on is unsaturated (aromatic/double/triple)
                jbond = self.molecule.GetBondBetweenAtoms(atom_indx[0], atom_indx[1])
                if bool(jbond):
                    if 'SINGLE' in str(jbond.GetBondType()):
                        # preserve the double bond geometry in petadiene
                        if ring_indx == 0:
                            self.molecule.AddBond(atom_indx[0], num_atom, Chem.BondType.DOUBLE) # the order of single vs double is important
                            self.molecule.AddBond(atom_indx[-1], num_atom + 2, Chem.BondType.SINGLE) # 2 because 2 more atoms are left, go to the end
                        elif ring_indx == 1:
                            self.molecule.RemoveBond(atom_indx[0], atom_indx[1])
                            self.molecule.AddBond(atom_indx[0], atom_indx[1], Chem.BondType.DOUBLE)
                            self.molecule.AddBond(atom_indx[0], num_atom, Chem.BondType.SINGLE)
                            self.molecule.AddBond(atom_indx[-1], num_atom + 1,
                                              Chem.BondType.SINGLE) # drawing the cyclopentadiene helps to understand the numbering, remember CH2 is 0
                        elif ring_indx == 2:
                            self.molecule.AddBond(atom_indx[0], num_atom + 1, Chem.BondType.DOUBLE)
                            self.molecule.AddBond(atom_indx[-1], num_atom + 2,
                                              Chem.BondType.DOUBLE)
                    else:
                        if ring_indx == 0 or ring_indx == 1:
                            # add onto C1 or C2, build 2 single bonds!, i.e. just keep the untouched double bond in pentadiene, the other double bond becomes single bond/aromtized
                            self.molecule.AddBond(atom_indx[0], num_atom, Chem.BondType.SINGLE)
                            self.molecule.AddBond(atom_indx[-1], num_atom + 2 - ring_indx, # add to 0, +2; add to 1, +1
                                      Chem.BondType.SINGLE)
                        elif ring_indx == 2:
                            # slightly more complicated, here we have to remove a bond in the original system to make it single bond - else some chemicals, e.g. benzothiadiazole would not form
                            self.molecule.RemoveBond(atom_indx[0], atom_indx[1])
                            self.molecule.AddBond(atom_indx[0], atom_indx[1], Chem.BondType.SINGLE)
                            self.molecule.AddBond(atom_indx[0], num_atom + 1, Chem.BondType.DOUBLE)
                            self.molecule.AddBond(atom_indx[-1], num_atom + 2,
                                          Chem.BondType.DOUBLE)
            # note that len(atom_indx) == 3 (add rings onto 3 atoms or 2 bonds for cyclopentadiene) is chemically feasible but very rare (basically on naphtalene/hydronaphtalene), not included here, it should be doable via simple add aliphatic ring and then adding 1/2 double bond
        elif ring_size == 6:
            # for benzene
            for i in range(len(atom_indx)):
                # produce the partial ring by deleting the number of unneeded atoms, since it's aromatic it doesn't matter where to add onto the ring
                ring.RemoveAtom(0)
                # make the attaching bonds aromatic unless they are already so
                if i != (len(atom_indx) - 1):
                    # only modify the bonds of the molecule in range
                    jbond = self.molecule.GetBondBetweenAtoms(atom_indx[i], atom_indx[i+1])
                    if bool(jbond):
                        if not bool(jbond.GetIsAromatic()):
                            # changes the bond in the original molecule to aromatic before stitching the new ones in
                            self.molecule.RemoveBond(atom_indx[i], atom_indx[i + 1])
                            self.molecule.AddBond(atom_indx[i], atom_indx[i + 1], Chem.BondType.AROMATIC)
            num_atom = self.molecule.GetNumAtoms()
            if len(atom_indx) == 6:
                # if this ring is merely a 'closure' of a ring with existing atoms, e.g. benzyl-2-methylbenzene
                if bool(self.molecule.GetBondBetweenAtoms(atom_indx[0], atom_indx[-1])):
                    self.molecule.AddBond(atom_indx[0], atom_indx[-1], Chem.BondType.AROMATIC)
            else:
                # stitch molecule and the new ring together
                self.molecule = Chem.RWMol(Chem.CombineMols(self.molecule, ring))
                # make new bonds
                self.molecule.AddBond(atom_indx[0], num_atom, Chem.BondType.AROMATIC)
                self.molecule.AddBond(atom_indx[-1], num_atom + ring_size - (1 + len(atom_indx)), Chem.BondType.AROMATIC)
        else:
            raise Exception('other ring sizes/features currently not supported')
        Chem.SanitizeMol(self.molecule)
        return None

    def random_walk(self, length, step=0): # gives an option of restarting at step=int
        while step < length:
            if self.molecule.GetNumAtoms() == 0:
                # create an atom/ring for intialization
                if random.choice([0, 1]) == 0:
                    atom_type = random.choice(Atoms[0:2]) # initialize with C, N, O only
                    self.add_atom(new_atom=int(atom_type))
                else:
                    aromatic = bool(random.getrandbits(1))
                    if aromatic:
                        ring_size = random.choice([5, 6])
                        self.add_ring(ring_size=ring_size, aromatic=aromatic)
                    else:
                        ring_size = np.random.choice([3, 4, 5, 6, 7, 8, 9])
                        self.add_ring(ring_size=ring_size, aromatic=aromatic)
                step = 1
            else:
                if self.molecule.GetNumAtoms() > 1:
                    action = random.choices(self.actions, weights=(20, 5, 2, 5, 2, 20))[0]
                else:
                    action = random.choice([self.add_atom, self.add_ring])
                prev_molecule = copy(self.molecule)
                try:
                    if any(method in action.__name__ for method in ['add_atom', 'edit_atom', 'delete_atom']): # actions requiring the generation of 1 atom_indx
                        atom_indx = [random.randint(0, self.molecule.GetNumAtoms(onlyExplicit=True)-1)]

                        # to not allow deleting atoms in a ring - temporarily
                        #if action.__name__ == 'delete_atom':
                        #    if self.molecule.GetAtomWithIdx(atom_indx[0]).IsInRing():
                        #        action=random.choice([self.add_atom, self.edit_atom])
                            #while self.molecule.GetAtomWithIdx(atom_indx[0]).IsInRing(): #doesn't work if it's only rings
                            #    atom_indx = [random.randint(0, self.molecule.GetNumAtoms(onlyExplicit=True) - 1)]

                        atom_type = int(random.choices(Atoms, weights=(25, 10, 10, 2, 1, 1, 1))[0])
                        action(atom_indx, atom_type)

                    elif any(method in action.__name__ for method in ['edit_bond', 'delete_bond']): # actions requiring the generaiton of 2 atom_indx
                        first_atom = random.randint(0, self.molecule.GetNumAtoms(onlyExplicit=True) - 1)
                        second_atom = random.choice(self.molecule.GetAtomWithIdx(first_atom).GetNeighbors()).GetIdx()
                        atom_indx = [first_atom, second_atom]
                        # this ensures the second atom is bonded to the first
                        if 'edit_bond' in action.__name__:
                            #print('edit_bond')
                            new_bond = random.choice(range(1, 3))
                            action(atom_indx, new_bond)
                            #print('edit_bond done')
                        elif 'delete_bond' in action.__name__:
                            #print('delete_bond')
                            # to not allow picking bonds in a ring - temporarily
                            #if self.molecule.GetBondBetweenAtoms(atom_indx[0], atom_indx[1]).IsInRing():
                            #    action=self.edit_bond
                            #    new_bond = random.choice(range(1, 3))
                            #    action(atom_indx, new_bond)
                                # while self.molecule.GetBondBetweenAtoms(atom_indx[0], atom_indx[1]).IsInRing():
                                #     first_atom = random.randint(0, self.molecule.GetNumAtoms(onlyExplicit=True) - 1)
                                #     second_atom = random.choice(
                                #         self.molecule.GetAtomWithIdx(first_atom).GetNeighbors()).GetIdx()
                                #     atom_indx = [first_atom, second_atom]
                            #else:
                            action(atom_indx)
                    else:
                        # add a ring
                        aromatic = bool(random.getrandbits(1))
                        if self.molecule.GetNumAtoms() <= 5:
                            atom_indx_length = random.choice(range(1, 2))
                        else: atom_indx_length = random.choice(range(1, 5))
                        # this ensures all atoms chosen are bonded to others, but it prevents rare cases of, e.g. joining the tails of a macrocycle
                        atom_indx = [random.randint(0, self.molecule.GetNumAtoms(onlyExplicit=True) - 1)] # get the first atom
                        for i in range(atom_indx_length-1): atom_indx.append(random.choice(self.molecule.GetAtomWithIdx(atom_indx[-1]).GetNeighbors()).GetIdx())
                        # alternatively the below version is slower/produces more unfeasible options but also offers more versality (doesn't rely on choosing the first atoms right)
                        # while atom_indx_length > len(atom_indx): # find others that are neighbors to the list
                        #     random_atom = random.randint(0, self.molecule.GetNumAtoms(onlyExplicit=True) - 1)
                        #     if random_atom not in atom_indx:
                        #         current_neighbors = self.molecule.GetAtomWithIdx(random_atom).GetNeighbors()
                        #         for neighbor in current_neighbors:
                        #             if neighbor.GetIdx() in atom_indx:
                        #                 atom_indx.append(random_atom)
                        if aromatic:
                            ring_size = random.choice(range(5, 6))
                            if ring_size == 5:
                                ring_indx = random.choice(range(3))
                                action(atom_indx, ring_size, aromatic, ring_indx)
                            else: action(atom_indx, ring_size, aromatic)
                        else:
                            ring_size = random.choice(range(3, 9))
                            action(atom_indx, ring_size, aromatic)
                    Chem.SanitizeMol(self.molecule)
                    step += 1
                    self.success += 1
                    prev_molecule = None
                except Exception as e:
                    #print (e)
                    #PrintException()
                    self.molecule = copy(prev_molecule)
                    self.error += 1
                    continue
            #print('action success', step, Chem.MolToSmiles(self.molecule))

    def _optimize(self):
        assert not "h" in set([atom.GetSymbol().lower() for atom in self.molecule.GetAtoms()]), "can't optimize molecule with h"
        Chem.AddHs(self.molecule)
        AllChem.EmbedMolecule(self.molecule)
        AllChem.MMFFOptimizeMolecule(self.molecule)
        Chem.RemoveHs(self.molecule)

    def mol(self):
        # for consistency as current reward functions uses molecule.mol; to be depreceated
        if self.molecule.GetNumAtoms() >= 1:
            return self.molecule.GetMol()
        else
            return None
        
    # add_bond is useful for fixing up fragments, but difficult to determine how it is feasible.
    # for nested ring, this is fixed by the add_ring option

    def aromatize_ring(self, ring):
        # todo makes a given ring aromatic and planar
        # technically, for benzene, this is the same as calling add_ring(aromatic, ring_size=6)
        return None

    def dearomatize_ring(self, ring):
        # todo makes a given ring aliphatic and nonplanar
        return None

    def delete_ring(self, ring):
        # todo
        return None

