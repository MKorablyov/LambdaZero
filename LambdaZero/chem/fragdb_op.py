import os,time,logging,itertools
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
from .. import chem
from matplotlib import pyplot as plt


class FragDB:
    def __init__(self):
        # raw fragments
        self.frag_names = {}
        self.frag_counts = []
        self.frag_elem = []
        self.frag_r = []
        self.frag_symmgroups = {}        # {frag_name: ig_table}

        # unique fragments
        self.ufrag_names = {}            # {ufrag_name: index} # ufrag_name = {frag_name}_{symmetry_group}
        self.ufrag_smis = []
        self.ufrag_counts = []
        self.ufrag_stem = []
        self.ufrag_r = []

    def add_frag(self, frag_name, frag_elem, frag_r):
        frag_natm = len(frag_elem)
        # if the fragment is new, initialize empty holders
        if not frag_name in self.frag_names:
            frag_idx = len(self.frag_names)
            self.frag_names[frag_name] = frag_idx
            self.frag_counts.append(0)
            self.frag_elem.append(frag_elem)
            self.frag_r.append(np.zeros([frag_natm,4]))
        # add fragments
        frag_idx = self.frag_names[frag_name]
        self.frag_counts[frag_idx] += 1
        assert self.frag_elem[frag_idx] == frag_elem, "fragment's order of elements does not repeat"

        # add r-groups in a sparse format
        if len(frag_r) > 0:
            frag_r_updates = np.zeros([frag_natm, 4])
            for atmid, count in Counter(frag_r).items():
                frag_r_updates[atmid,count-1] = 1
            self.frag_r[frag_idx] = self.frag_r[frag_idx] + frag_r_updates
        else:
            pass # molecule consisted of a single fragment; no updates to the R-groups

    def add_frags(self, jun_bonds, frag_names, frag_elem):
        # for each of the groups find a list of fragment bonds, and R-groups
        frag_r = [[] for _ in range(len(jun_bonds) + 1)]
        for bond in jun_bonds:
            frag_r[bond[0]].append(bond[2])
            frag_r[bond[1]].append(bond[3])
        nfrags = len(frag_names)
        for i in range(nfrags):
            self.add_frag(frag_names[i], frag_elem[i], frag_r[i])

    def add_mol(self, mol, frags_generic, decomposition="rota_murcko"):
        frag_elem, frag_coord, _, jun_bonds, frag_names = \
            chem.fragment_molecule(mol, frags_generic, decomposition=decomposition)

        self.add_frags(jun_bonds, frag_names, frag_elem)
        return frag_elem, jun_bonds, frag_names

    def make_ufrags(self,verbose=False):
        "generate the new database of unique fragments "
        passed_nfrag = 0
        broken_nfrag = 0
        for frag_name in self.frag_names:
            # compute isometry groups of a fragment
            try:
                frag_mol = Chem.MolFromSmiles(frag_name)
                frag_symm = chem.compute_isometry(frag_mol)
            except Exception as e:
                # todo log errors
                broken_nfrag += 1
                continue
            passed_nfrag += 1
            self.frag_symmgroups[frag_name] = frag_symm

            # find (possible) bonds of a fragment
            frag_r = self.frag_r[self.frag_names[frag_name]]
            atm_bonded = frag_r.sum(1) > 0
            frag_symm_bonded = np.unique(frag_symm[atm_bonded])
            frag_natm = frag_r.shape[0]

            for symm_group in frag_symm_bonded:
                # add ufrag name
                ufrag_name = frag_name + "_" + str(symm_group)
                ufrag_idx = len(self.ufrag_names)
                self.ufrag_names[ufrag_name] = ufrag_idx
                self.ufrag_smis.append(frag_name)
                # ufrag count
                ufrag_count = frag_r[(frag_symm == symm_group)].sum(0)
                ufrag_count = (ufrag_count * np.array([1,2,3,4])).sum()
                self.ufrag_counts.append(ufrag_count)
                # ufrag_r
                # todo: add all R-groups > count except that one
                stem = np.where((frag_symm == symm_group))[0][0]
                r_mask = np.ones(frag_natm)
                r_mask[stem] = 0.
                self.ufrag_stem.append(stem)
                self.ufrag_r.append(frag_r * r_mask[:,None])

        total_ufrag = (np.concatenate(self.frag_r,axis=0) * np.array([[1,2,3,4]])).sum()
        print("passed frags:", passed_nfrag, "broken frags:", broken_nfrag,
              "passed ufrag count", np.sum(self.ufrag_counts), "total ufrag count", total_ufrag)
        if verbose:
            plt.plot(-np.sort(-np.asarray(self.ufrag_counts)))
            plt.yscale('log')
            plt.xlabel("ufrag id")
            plt.ylabel("ufrag count")
            plt.show()
        return None

    def filter_ufrags(self, cutoff_count=250, verbose=True):
        "deletes most of the uni"
        # unique fragments
        sel_ufrag_names = {}
        sel_ufrag_smis = []
        sel_ufrag_counts = []
        sel_ufrag_r = []
        for ufrag_name in self.ufrag_names:
            ufrag_idx = self.ufrag_names[ufrag_name]
            if self.ufrag_counts[ufrag_idx] >= cutoff_count:
                sel_ufrag_smis.append(self.ufrag_smis[ufrag_idx])
                sel_ufrag_idx = len(sel_ufrag_counts)
                sel_ufrag_names[ufrag_name] = sel_ufrag_idx
                sel_ufrag_counts.append(self.ufrag_counts[ufrag_idx])

                ufrag_r = self.ufrag_r[ufrag_idx]
                ufrag_r = np.fliplr(np.cumsum(np.fliplr(ufrag_r), axis=1))
                ufrag_r = np.concatenate([np.array([self.ufrag_stem[ufrag_idx]]),
                                          np.where(ufrag_r[:,3] >= cutoff_count)[0],
                                          np.where(ufrag_r[:,2] >= cutoff_count)[0],
                                          np.where(ufrag_r[:,1] >= cutoff_count)[0],
                                          np.where(ufrag_r[:,0] >= cutoff_count)[0]])

                sel_ufrag_r.append(ufrag_r)
        # reorder keys values alphabetically
        sel_ufrag_names, order = zip(*sel_ufrag_names.items())
        sel_ufrag_counts = np.asarray(sel_ufrag_counts)[np.asarray(order)]
        sel_ufrag_r = np.asarray(sel_ufrag_r)[np.asarray(order)]

        print("num ufrag selected", len(sel_ufrag_counts),
              "total ufrag count", np.sum(self.ufrag_counts),
              "sel_ufrag_count", np.sum(sel_ufrag_counts))
        print("selected ufrags:", sel_ufrag_names)
        return sel_ufrag_names, sel_ufrag_smis, sel_ufrag_counts, sel_ufrag_r


    def draw_ufrag(self,draw_folder):
        for ufrag_name,idx in self.ufrag_names.items():
            out_file = str(idx) + "_" + str(int(self.ufrag_counts[idx])) + "_" + ufrag_name
            chem.draw_frag(ufrag_name.split("_")[0],[int(ufrag_name.split("_")[-1])], draw_folder, out_file)

    def get_ufrag(self,frag_name,frag_bond):
        frag_symm = self.frag_symmgroups[frag_name]
        symm_group = frag_symm[frag_bond] # fixme should it be 4 ???
        ufrag_name = frag_name + "_" + str(symm_group)
        ufrag_idx = self.ufrag_names[ufrag_name]
        ufrag_count = self.ufrag_counts[ufrag_idx]
        return ufrag_idx,ufrag_count

    def get_mol_ufrag(self, mol, frags_generic, decomposition="rota_murcko", elem_asnum=True):
        # break molecule into fragments
        frag_elems, frag_coords, _, jun_bonds, frag_names = chem.fragment_molecule(mol, frags_generic, decomposition)
        # find every fragment in the database
        ufrag_idxs = []
        flat_jun_bonds = np.concatenate([np.stack([jun_bonds[:,0], jun_bonds[:,2]],1),
                                         np.stack([jun_bonds[:,1], jun_bonds[:,3]],1)],axis=0)
        for bond in flat_jun_bonds:
            ufrag_idxs.append(self.get_ufrag(frag_names[bond[0]], bond[1])[0])
        # convert elements to atomic numbers
        if elem_asnum:
            _frag_elems = []
            for frag_elem in frag_elems:
                _frag_elems.append(np.asarray([chem.atomic_numbers[e] for e in frag_elem],dtype=np.float32))
            frag_elems = _frag_elems
        return frag_elems, frag_coords, jun_bonds, ufrag_idxs

    # def _select_fragments(self,num_frags,minfreq_r):
    #     # fixme count_fragments ???
    #     db_num_frags = len(self.frag_names)
    #
    #     # create a filter for the database of fragments
    #     all_frag_r = []
    #     frag_r_mask = []
    #     for frag_idx in range(db_num_frags):
    #         frag_natm = len(self.frag_elem[frag_idx])
    #
    #         # apply filters and convert R-groups to the sparse format
    #         frag_r = self.frag_r[frag_idx] / self.frag_counts[frag_idx]
    #         frag_r = np.fliplr(np.cumsum(np.fliplr(frag_r),axis=1)) > minfreq_r
    #         frag_r = np.concatenate([np.ones([frag_natm,1]), np.asarray(frag_r,dtype=np.float)],1)
    #         frag_r = np.argmax(frag_r * np.array([[1,2,3,4,5]]),axis=1)
    #         nonzero_idx = np.argwhere(frag_r)[:,0]
    #         frag_r = np.stack([nonzero_idx,frag_r[nonzero_idx]],axis=1)
    #         # frag_r =  np.repeat(nonzero_idx, frag_r[nonzero_idx])
    #         all_frag_r.append(frag_r)
    #         if len(frag_r): frag_r_mask.append(True),
    #         else: frag_r_mask.append(False)
    #
    #     sel_mask = np.asarray(frag_r_mask, dtype=np.float) * np.asarray(self.frag_counts)
    #     thr = -np.sort(-sel_mask)[num_frags]
    #     sel_idxs = np.arange(db_num_frags)[sel_mask > thr]
    #     print("selecting fragments with an occurence threshhold of", thr)
    #
    #     # take fragments from the database
    #     (frag_names,frag_idxs) = zip(*self.frag_names.items())
    #     frag_names = np.asarray(frag_names)[np.asarray(frag_idxs)]
    #     sel_frag_names = []
    #     sel_frag_counts = []
    #     sel_frag_elem = []
    #     # sel_frag_bonds = []
    #     sel_frag_r = []
    #     for frag_idx in sel_idxs:
    #         sel_frag_names.append(frag_names[frag_idx])
    #         sel_frag_counts.append(self.frag_counts[frag_idx])
    #         sel_frag_elem.append(self.frag_elem[frag_idx])
    #         # sel_frag_bonds.append(self.frag_bonds[frag_idx])
    #         sel_frag_r.append(all_frag_r[frag_idx])
    #     return sel_frag_names,sel_frag_counts,sel_frag_elem,sel_frag_r


    # def build_fragspace(self, nfrags, minfreq_r, fragspace_path):
    #     "build a fragment space of this database"
    #     frag_names, frag_counts, frag_elems, frag_rs = self._select_fragments(nfrags,minfreq_r)
    #     fragspace_index = []
    #     for frag_idx in range(nfrags):
    #         record = save_frag(frag_names[frag_idx], frag_rs[frag_idx],frag_names,frag_rs,fragspace_path)
    #         fragspace_index.append(record)
    #     print ("finished building the fragspace")
    #     fragspace_nfrags = np.sum(np.asarray([r.shape[0] > 0 for r in fragspace_index], dtype=np.int32))
    #     print ("initial number of fragments:", nfrags, "converted:", fragspace_nfrags)
    #     nr = np.sum(np.asarray([np.sum(frag_r[:,1]) for frag_r in frag_rs],dtype=np.int32))
    #     fragspace_nr = np.sum(np.asarray([r.shape[0] for r in fragspace_index], dtype=np.int32))
    #     print ("initial number of r-groups:", nr, "converted:", fragspace_nr)
    #     fragspace_index = np.asarray(np.concatenate(fragspace_index,axis=0),dtype=np.str)
    #     np.save(os.path.join(fragspace_path,"fragspace_index.npy"), fragspace_index)

    def save_state(self, fragdb_path):
        "write this object to the disk"
        if not os.path.exists(fragdb_path): os.makedirs(fragdb_path)
        nr = len(self.frag_names), len(self.frag_counts), len(self.frag_elem), len(self.frag_r)
        assert all(np.asarray(nr) == nr[0]), "database columns are not of the same length"
        np.save(os.path.join(fragdb_path, "frag_names.npy"), self.frag_names)
        np.save(os.path.join(fragdb_path, "frag_counts.npy"), self.frag_counts)
        np.save(os.path.join(fragdb_path, "frag_elem.npy"), self.frag_elem)
        np.save(os.path.join(fragdb_path, "frag_r.npy"), self.frag_r)
        np.save(os.path.join(fragdb_path, "frag_symmgroups.npy"), self.frag_symmgroups)
        assert len(self.ufrag_names) == len(self.ufrag_counts), "broken database of ufragments"
        np.save(os.path.join(fragdb_path, "ufrag_names.npy"), self.ufrag_names)
        np.save(os.path.join(fragdb_path, "ufrag_counts.npy"), self.ufrag_counts)
        print('saved database state nfrag:', len(self.frag_names))

    def load_state(self, fragdb_path, ufrag):
        "write this object to the disk"
        self.frag_names = np.load(os.path.join(fragdb_path, "frag_names.npy"),allow_pickle=True).item()
        self.frag_counts = np.load(os.path.join(fragdb_path, "frag_counts.npy"))
        self.frag_elem = np.load(os.path.join(fragdb_path, "frag_elem.npy"),allow_pickle=True)
        self.frag_r = np.load(os.path.join(fragdb_path, "frag_r.npy"),allow_pickle=True)
        nr = len(self.frag_names.keys()), len(self.frag_counts), len(self.frag_elem), len(self.frag_r)
        assert all(np.asarray(nr) == nr[0]), "database columns are not of the same length"
        if ufrag:
            self.frag_symmgroups = np.load(os.path.join(fragdb_path, "frag_symmgroups.npy"),allow_pickle=True).item()
            self.ufrag_names = np.load(os.path.join(fragdb_path, "ufrag_names.npy"),allow_pickle=True).item()
            self.ufrag_counts = np.load(os.path.join(fragdb_path, "ufrag_counts.npy"))
            assert len(self.ufrag_names) == len(self.ufrag_counts), "broken database of ufragments"
        print('loaded database state nfrag:', len(self.frag_names))