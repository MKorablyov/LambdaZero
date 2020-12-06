import contextlib
import logging
import os
import random
import string
import subprocess
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

from LambdaZero.chem.chimera_op import _add_hydrogens_and_compute_gasteiger_charges_with_chimera
from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

rdBase.DisableLog('rdApp.error')

import pandas as pd
from rdkit import DataStructs

import torch
from torch_geometric.data import (Data)
from torch_sparse import coalesce

atomic_numbers = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
                  "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
                  "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,
                  "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
                  "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56}


def compute_isometry(mol):
    ":return [num_atoms] isometric group "
    isom_groups = list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    return np.asarray(isom_groups, dtype=np.int32)


def find_rota_murcko_bonds(mol, sanitize=False):
    """ Finds rotatable bonds (first) and Murcko bonds (second)
    :param mol:
    :return:
    """
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError("error in sanitization") from e

    # r_groups decomposition
    rotatable = Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')
    rota_bonds = mol.GetSubstructMatches(rotatable)
    r_bonds = [((x, y), (0, 0)) for x, y in rota_bonds]
    r_groups = BRICS.BreakBRICSBonds(mol, r_bonds)
    r_atmids = list(Chem.GetMolFrags(r_groups))
    r_groups = list(Chem.GetMolFrags(r_groups, asMols=True))

    # Murcko decomposition
    murcko_bonds = []  # murcko bonds
    for rgroup_num in range(len(r_groups)):
        m_core = MurckoScaffold.GetScaffoldForMol(r_groups[rgroup_num])
        if m_core.GetNumAtoms() > 0:
            # get all bonds in this fragment
            rg_bonds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in r_groups[rgroup_num].GetBonds()]
            #print ([bond.GetBondType() for bond in r_groups[rgroup_num].GetBonds()])
            rg_bonds = np.asarray(rg_bonds)
            # remove bonds that lead to dummy atoms added by BreakBRICSBonds
            dummy_atoms = np.asarray(r_atmids[rgroup_num]) >= mol.GetNumAtoms()
            dummy_bonds = dummy_atoms[rg_bonds.reshape([-1])].reshape([-1, 2])
            dummy_bonds = np.logical_or(dummy_bonds[:, 0], dummy_bonds[:, 1])
            rg_bonds = rg_bonds[np.logical_not(dummy_bonds)]
            # filter out only the ones that connect core with something outside of the core
            mcore_atmid = np.asarray(r_groups[rgroup_num].GetSubstructMatch(m_core))
            m_bonds = np.reshape(np.in1d(rg_bonds, mcore_atmid), [-1, 2])
            m_bonds = rg_bonds[np.logical_xor(m_bonds[:, 0], m_bonds[:, 1])]
            # return bond-numbering to the one for a whole molecule
            m_bonds = np.asarray(r_atmids[rgroup_num])[np.reshape(m_bonds, [-1])]
            m_bonds = np.reshape(m_bonds, [-1, 2])
            [murcko_bonds.append(m_bond) for m_bond in m_bonds]

    rota_bonds = np.reshape(np.asarray(rota_bonds, dtype=np.int64), [-1, 2])
    murcko_bonds = np.reshape(np.asarray(murcko_bonds, dtype=np.int64), [-1, 2])
    return rota_bonds, murcko_bonds


def break_on_bonds(mol, jun_bonds, frags_generic):
    """ Breaks molecule into fragments
    :param mol: molecule
    :param junction_bonds: junction_bonds [n_junction_bonds,4] bond = [frag0, frag1, atm0, atm1]
    :return:
    """
    # fixme break ties
    # todo bondtypes (?)
    # todo assert jbonds num atoms
    n_junct = jun_bonds.shape[0]
    # break bonds of the molecule; get fragments
    Chem.Kekulize(mol, clearAromaticFlags=False)
    emol = Chem.EditableMol(mol)
    [emol.RemoveBond(int(jun_bonds[i, 0]), int(jun_bonds[i, 1])) for i in range(n_junct)]
    frags = list(Chem.GetMolFrags(emol.GetMol(), asMols=True, sanitizeFrags=True))

    if frags_generic:
        frags = [MurckoScaffold.MakeScaffoldGeneric(frag) for frag in frags]
    assert len(frags) == n_junct + 1, "internal error: molecule did not break into the right number of fragments"
    # get a frag_name for each of the groups
    frag_names = [Chem.MolToSmiles(f) for f in frags]
    # reorder atom numbers to follow exactly the SMILES string we just took
    frag_canonord = [frags[i].GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
                     for i in range(n_junct + 1)]
    mol_canonord = list(np.asarray(frag) for frag in Chem.GetMolFrags(emol.GetMol()))
    mol_canonord = [mol_canonord[i][np.asarray(frag_canonord[i])] for i in range(n_junct+1)]
    mol_canonord = np.concatenate(mol_canonord, 0)
    frags = [Chem.RenumberAtoms(frags[i], frag_canonord[i]) for i in range(n_junct + 1)]
    mol = Chem.RenumberAtoms(mol, [int(idx) for idx in mol_canonord])
    frag_startidx = np.concatenate([[0], np.cumsum([frag.GetNumAtoms() for frag in frags])], 0)[:-1]
    # get fragment elements frag_elem and frag_coord
    frag_atoms = [frags[i].GetAtoms() for i in range(n_junct+1)]
    frag_elem = []
    frag_coord = []
    for i in range(n_junct +1):
        frag_natm = len(frag_atoms[i])
        frag_elem.append([frag_atoms[i][j].GetSymbol() for j in range(frag_natm)])
        frag_coord.append(np.asarray([frags[i].GetConformer(0).GetAtomPosition(j) for j in range(frag_natm)]))
    # find junction/fragment bonds for each of the groups
    bonds = np.asarray([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in list(mol.GetBonds())]).reshape([-1, 2])
    bond_frags = (np.searchsorted(frag_startidx, bonds, side='right') - 1)
    bond_atms = bonds - frag_startidx[bond_frags]
    bonds = np.concatenate([bond_frags, bond_atms], 1)
    is_junction = [bond_frags[:, 0] != bond_frags[:, 1]][0]
    jun_bonds = bonds[is_junction]
    assert jun_bonds.shape[0] == n_junct, "internal error in fragmentation"
    frag_bonds = [[] for _ in range(n_junct + 1)]
    for bond in bonds[np.logical_not(is_junction)]:
        frag_bonds[bond[0]].append(bond[2:])
    return jun_bonds, frag_names, frag_elem, frag_coord, frag_bonds


def mol_from_frag(jun_bonds, frags=None, frag_smis=None, coord=None, optimize=False):
    "joins 2 or more fragments into a single molecule"
    jun_bonds = np.asarray(jun_bonds)
    #if jun_bonds.shape[0] == 0: jun_bonds = np.empty([0,4])
    if frags is not None:
        pass
    elif frags is None and frag_smis is not None:
        frags = [Chem.MolFromSmiles(frag_name) for frag_name in frag_smis]
    else:
        raise ValueError("invalid argument either frags or frags smis should be not None")
    if len(frags) == 0: return None, None
    nfrags = len(frags)
    # combine fragments into a single molecule
    mol = frags[0]
    for i in np.arange(nfrags-1)+1:
        mol = Chem.CombineMols(mol, frags[i])
    # add junction bonds between fragments
    frag_startidx = np.concatenate([[0], np.cumsum([frag.GetNumAtoms() for frag in frags])], 0)[:-1]

    if jun_bonds.size == 0:
        mol_bonds = []
    else:
        mol_bonds = frag_startidx[jun_bonds[:, 0:2]] + jun_bonds[:, 2:4]

    emol = Chem.EditableMol(mol)

    [emol.AddBond(int(bond[0]), int(bond[1]), Chem.BondType.SINGLE) for bond in mol_bonds]
    mol = emol.GetMol()
    atoms = list(mol.GetAtoms())

    def _pop_H(atom):
        nh = atom.GetNumExplicitHs()
        if nh > 0: atom.SetNumExplicitHs(nh-1)

    [(_pop_H(atoms[bond[0]]), _pop_H(atoms[bond[1]])) for bond in mol_bonds]
    #print([(atom.GetNumImplicitHs(), atom.GetNumExplicitHs(),i) for i,atom in enumerate(mol.GetAtoms())])
    Chem.SanitizeMol(mol)
    # create and optimize 3D structure
    if optimize:
        assert not "h" in set([atm.GetSymbol().lower() for atm in mol.GetAtoms()]), "can't optimize molecule with h"
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.RemoveHs(mol)
    return mol, mol_bonds


def fragment_molecule(mol, frags_generic, decomposition):
    """ Fragments a whole molecule and adds to the database.
    :param mol:
    :param frags_generic:
    :param add_frags:
    :return:
    """
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError("error in sanitization") from e
    try:
        if decomposition == "rota_murcko":
            rota_bonds, murcko_bonds = find_rota_murcko_bonds(mol)
            jun_bonds = np.concatenate([rota_bonds, murcko_bonds], axis=0)
        else:
            raise NotImplementedError("decomposition")
        jun_bonds, frag_names, frag_elem, frag_coord, frag_bonds = break_on_bonds(mol, jun_bonds, frags_generic)
    except Exception as e:
        raise ValueError("error in fragmenting") from e
    return frag_elem, frag_coord, frag_bonds, jun_bonds, frag_names


def draw_frag(frag_name, r_idx, out_path, out_file=None):
    "0th r_idx should be the actual stem of the group"
    # fixme - this function should be generic
    # create a molecule object
    fragsp_name = frag_name + "_" + str(r_idx[0])
    frag = Chem.MolFromSmiles(frag_name)
    frag_natm = len(frag.GetAtoms())
    # add groups to this molecule for each of the R-groups of interest
    efrag = Chem.EditableMol(frag)
    dummy = Chem.Atom(0)
    dummy.SetNoImplicit(True)
    efrag.AddAtom(dummy)
    efrag.AddBond(frag_natm, int(r_idx[0]), Chem.rdchem.BondType.UNSPECIFIED)
    frag_proto = efrag.GetMol()
    Chem.AddHs(frag_proto)
    Chem.SanitizeMol(frag_proto)
    # save image
    if not os.path.exists(out_path): os.makedirs(out_path)
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400, 400)
    opts = drawer.drawOptions()
    for i in range(frag_proto.GetNumAtoms()):
        opts.atomLabels[i] = frag_proto.GetAtomWithIdx(i).GetSymbol() + str(i)
    Chem.rdDepictor.Compute2DCoords(frag_proto)
    drawer.DrawMolecule(frag_proto, highlightAtoms=[int(r) for r in r_idx])
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg.replace('svg:', '')

    if out_file is None:
        out_path = os.path.join(out_path, fragsp_name + ".svg")
    else:
        out_path = os.path.join(out_path, out_file + ".svg")
    svg_file = open(out_path, "w")
    svg_file.write(svg)
    svg_file.close()
    logging.info("saved image", os.path.join(out_path, fragsp_name + ".svg"))
    return


def build_mol(smiles=None, num_conf=1, minimize=False, noh=True, charges=True):
    # todo: things other than SMILES
    # fixme: return numpy array and not list
    # todo: select min energy?
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    # create and optimize 3D structure
    if num_conf > 0:
        assert not "h" in set([atm.GetSymbol().lower() for atm in mol.GetAtoms()]), "can't optimize molecule with h"
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf)
        if minimize:
            [AllChem.MMFFOptimizeMolecule(mol, confId=i) for i in range(num_conf)]
        if charges:
            AllChem.ComputeGasteigerCharges(mol)
            #print("charge 0",float(mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')))
        if noh:
            mol = Chem.RemoveHs(mol)
    # get elem, get coord
    elem = [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    coord = [np.asarray([mol.GetConformer(j).GetAtomPosition(i) for i in range(len(elem))]) for j in range(num_conf)]
    coord = np.asarray(np.stack(coord, axis=0), dtype=np.float32).tolist()
    return mol, elem, coord


def _gen_mol2(smi, mol_name, outpath, chimera_bin, charge_method, num_conf):
    # generate num_conf 3D conformers from smiles
    mol = Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(mol)
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf)
    [AllChem.MMFFOptimizeMolecule(mol_h, confId=i) for i in range(num_conf)]
    mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
    # choose minimum energy conformer
    mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy() for i in range(num_conf)])
    mol = Chem.RemoveHs(mol_h)
    # save file in .mol format
    mol_file = os.path.join(outpath, mol_name + ".mol")
    print(Chem.MolToMolBlock(mol, confId=int(mi)), file=open(mol_file, 'w+'))
    # add hydrogens and compute gasteiger charges in Chimera
    mol2_file = os.path.join(outpath, mol_name + ".mol2")
    _add_hydrogens_and_compute_gasteiger_charges_with_chimera(mol_file, charge_method, chimera_bin, mol2_file)

    return mol2_file


class Dock_smi:
    def __init__(self,
                 outpath,
                 chimera_dir,       #
                 dock6_dir,         #
                 docksetup_dir,     #
                 cleanup=True,
                 trustme=False,
                 gas_charge=True,
                 rec_site_file="2_site/selected_spheres.sph",
                 grid_prefix="3_grid/grid",
                 dock_in_template="4_dock/anchor_and_grow.in",
                 vdw_defn_file="parameters/vdw_AMBER_parm99.defn",
                 flex_defn_file="parameters/flex.defn",
                 flex_drive_file="parameters/flex_drive.tbl"):

        # output path
        self.outpath = outpath
        if gas_charge:
            self.charge_method = "gas"
        else:
            self.charge_method = "am1-bcc"

        if not trustme:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
        # chimera bin
        self.chimera_bin = os.path.join(chimera_dir, "bin/chimera")
        if not trustme:
            assert os.path.exists(self.chimera_bin), "can't find chimera bin " + self.chimera_bin
        # dock6 bin and parameter files
        self.dock6_bin = os.path.join(dock6_dir, "bin/dock6")
        if not trustme:
            assert os.path.exists(self.dock6_bin), "can't find dock6 bin " + self.dock6_bin
        self.vdw_defn_file = os.path.join(dock6_dir, vdw_defn_file)
        if not trustme:
            assert os.path.exists(self.vdw_defn_file), "can't find vdw_def_file " + self.vdw_defn_file
        self.flex_defn_file = os.path.join(dock6_dir, flex_defn_file)
        if not trustme:
            assert os.path.exists(self.flex_defn_file), "can't find flex_dfn_file " + self.flex_defn_file
        self.flex_drive_file = os.path.join(dock6_dir, flex_drive_file)
        if not trustme:
            assert os.path.exists(self.flex_drive_file), "can't find dock6 bin " + self.flex_drive_file
        self.cleanup = cleanup
        # docking files
        self.rec_site_file = os.path.join(docksetup_dir, rec_site_file)
        if not trustme:
            assert os.path.exists(self.rec_site_file), "can't find rec site file " + self.rec_site_file
        self.grid_prefix = os.path.join(docksetup_dir, grid_prefix)
        self.dock_in_template = os.path.join(docksetup_dir, dock_in_template)
        if not trustme:
            assert os.path.exists(self.dock_in_template), "can't find rec site file " + self.dock_in_template

    def dock(self, smi, mol_name=None, molgen_conf=20):
        try:
            # generate random molecule name if needed
            if mol_name is None:
                mol_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))

            # do smiles conversion and docking
            mol2_file = _gen_mol2(smi, mol_name, self.outpath, self.chimera_bin, self.charge_method, num_conf=molgen_conf)
            gridscore, coord = self._dock_dock6(smi, mol2_file, mol_name,
                                                self.outpath, self.dock_in_template, self.dock6_bin)
        finally:
            if self.cleanup:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, mol_name + ".mol"))

                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, mol_name + ".mol2"))

                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, mol_name + "_dockin"))

                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, mol_name + "_dockout"))

                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, mol_name + "_scored.mol2"))

        return mol_name, gridscore, coord

    def _dock_dock6(self, smi, mol2file, mol_name, outpath, dock_pars_file, dock_bin):
        # write dockfile
        with open(dock_pars_file, 'r') as textfile: dock_pars_ = textfile.read()
        dock_pars = dock_pars_.format(mol2file, self.rec_site_file, self.grid_prefix,
                                      self.vdw_defn_file, self.flex_defn_file, self.flex_drive_file,
                                      os.path.join(outpath, mol_name))
        dock_pars_file = os.path.join(outpath, mol_name + "_dockin")
        print(dock_pars, file=open(dock_pars_file, "w+"), end='')

        # do docking
        dock_out_file = os.path.join(outpath, mol_name + "_dockout")
        dock_cmd = "{} -i {} -o {}".format(dock_bin, dock_pars_file, dock_out_file)
        process = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

        # parse dock energy
        with open(dock_out_file) as f:
            lines = f.read().splitlines()
        gridscores = [float(line[38:]) for line in lines if line.startswith("                          Grid_Score")]
        assert len(gridscores) == 1, "could not parse docking output - something happened while running dock binary"
        gridscore = gridscores[0]

        # parse dock coords
        mol2file_scored = os.path.join(self.outpath, mol_name + "_scored.mol2")
        mol = Chem.MolFromMol2File(mol2file_scored, sanitize=False, cleanupSubstructures=False)
        atoms = np.asarray([atm.GetSymbol().lower() for atm in mol.GetAtoms() if atm.GetSymbol().lower() != "h"])
        initmol = Chem.MolFromSmiles(smi)
        initatoms = np.asarray([atm.GetSymbol().lower() for atm in initmol.GetAtoms()])
        assert np.array_equal(atoms, initatoms), "atoms are not the same after parsing dock mol2"

        noh = np.asarray([atm.GetSymbol().lower() != "h" for atm in mol.GetAtoms()])
        noh = np.where(noh)[0]
        coord = np.asarray([mol.GetConformer(0).GetAtomPosition(int(idx)) for idx in noh])
        return gridscore, coord


class GenMolFile:
    def __init__(self, outpath, mgltools):
        self.outpath = outpath
        self.prepare_ligand4 = os.path.join(mgltools, "AutoDockTools/Utilities24/prepare_ligand4.py")

        os.makedirs(os.path.join(outpath, "sdf"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "mol2"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "pdbqt"), exist_ok=True)

    def __call__(self, smi, mol_name, num_conf):
        sdf_file = os.path.join(self.outpath, "sdf", f"{mol_name}.sdf")
        mol2_file = os.path.join(self.outpath, "mol2", f"{mol_name}.mol2")
        pdbqt_file = os.path.join(self.outpath, "pdbqt", f"{mol_name}.pdbqt")

        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf)
        for i in range(num_conf):
            AllChem.MMFFOptimizeMolecule(mol_h, confId=i)
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
        # choose minimum energy conformer
        mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy() for i in range(num_conf)])
        print(Chem.MolToMolBlock(mol_h, confId=int(mi)), file=open(sdf_file, 'w+'))
        os.system(f"obabel -isdf {sdf_file} -omol2 -O {mol2_file}")
        os.system(f"pythonsh {self.prepare_ligand4} -l {mol2_file} -o {pdbqt_file}")
        return pdbqt_file


class DockVina_smi:
    def __init__(self,
                 outpath,
                 mgltools_dir=os.path.join(programs_dir, "mgltools_x86_64Linux2_1.5.6"),
                 vina_dir=os.path.join(programs_dir, "vina"),
                 docksetup_dir=os.path.join(datasets_dir, "seh/4jnc"),
                 rec_file="4jnc.nohet.aligned.pdbqt",
                 bindsite=(-13.4, 26.3, -13.3, 20.013, 16.3, 18.5),
                 dock_pars="",
                 cleanup=True):

        self.outpath = outpath
        self.mgltools = os.path.join(mgltools_dir, "MGLToolsPckgs")
        self.vina_bin = os.path.join(vina_dir, "bin/vina")
        self.rec_file = os.path.join(docksetup_dir, rec_file)
        self.bindsite = bindsite
        self.dock_pars = dock_pars
        self.cleanup = cleanup

        self.gen_molfile = GenMolFile(self.outpath, self.mgltools)
        # make vina command
        self.dock_cmd = "{} --receptor {} " \
                        "--center_x {} --center_y {} --center_z {} " \
                        "--size_x {} --size_y {} --size_z {} "
        self.dock_cmd = self.dock_cmd.format(self.vina_bin, self.rec_file, *self.bindsite)
        self.dock_cmd += " --ligand {} --out {}"

        os.makedirs(os.path.join(self.outpath, "docked"), exist_ok=True)

    def dock(self, smi, mol_name=None, molgen_conf=20):
        mol_name = mol_name or ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        dockscore = float('nan')
        coord = None
        try:
            docked_file = os.path.join(self.outpath, "docked", f"{mol_name}.pdb")
            input_file = self.gen_molfile(smi, mol_name, molgen_conf)
            # complete docking query
            dock_cmd = self.dock_cmd.format(input_file, docked_file)
            dock_cmd = dock_cmd + " " + self.dock_pars
            # dock
            cl = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
            cl.wait()
            # parse energy
            with open(docked_file) as f:
                docked_pdb = f.readlines()
            if docked_pdb[1].startswith("REMARK VINA RESULT"):
                dockscore = float(docked_pdb[1].split()[3])
            else:
                raise Exception("Can't parse docking energy")
            # parse coordinates
            # TODO: fix order
            coord = []
            endmodel_idx = 0
            for idx, line in enumerate(docked_pdb):
                if line.startswith("ENDMDL"):
                    endmodel_idx = idx
                    break

            docked_pdb_model_1 = docked_pdb[:endmodel_idx]  # take only the model corresponding to the best energy
            docked_pdb_model_1_atoms = [line for line in docked_pdb_model_1 if line.startswith("ATOM") and line.split()[2][0] != 'H']  # ignore hydrogen
            coord.append([line.split()[-7:-4] for line in docked_pdb_model_1_atoms])
            coord = np.array(coord, dtype=np.float32)

        finally:
            if self.cleanup:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, "sdf", f"{mol_name}.sdf"))

                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, "mol2", f"{mol_name}.mol2"))

                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, "pdbqt", f"{mol_name}.pdbqt"))

                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.outpath, "docked", f"{mol_name}.pdb"))

        return mol_name, dockscore, coord

    def __del__(self):
        if self.cleanup:
            with contextlib.suppress(FileNotFoundError):
                os.rmdir(os.path.join(self.outpath, "sdf"))

            with contextlib.suppress(FileNotFoundError):
                os.rmdir(os.path.join(self.outpath, "mol2"))

            with contextlib.suppress(FileNotFoundError):
                os.rmdir(os.path.join(self.outpath, "pdbqt"))

            with contextlib.suppress(FileNotFoundError):
                os.rmdir(os.path.join(self.outpath, "docked"))


class ScaffoldSplit:
    def __init__(self, mols_smiles, incude_chirality=False):
        scaffolds = []
        for i in range(len(mols_smiles)):
            mol = Chem.MolFromSmiles(mols_smiles[i])
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=incude_chirality)
            scaffolds.append(scaffold)
        print(Counter(scaffolds))


def get_fp(mol, fp_length, fp_radiis, from_atoms=None):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
    """
    if mol is None:
        return np.zeros((fp_length * len(fp_radiis),), dtype=np.float32)
    if from_atoms is not None:
        from_atoms = [int(idx) for idx in from_atoms]

        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, r, fp_length, fromAtoms=from_atoms) for r in fp_radiis]

    else:
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, r, fp_length) for r in fp_radiis]

    fps_ = []
    for fp in fps:
        arr = np.zeros((1,))
        # ConvertToNumpyArray takes ~ 0.19 ms, while
        # np.asarray takes ~ 4.69 ms
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_.append(arr)
    fps = np.asarray(np.concatenate(fps_, axis=0), dtype=np.float32)
    return fps


class FPEmbedding_v2:
    def __init__(self, mol_fp_len, mol_fp_radiis, stem_fp_len, stem_fp_radiis):
        self.mol_fp_len = mol_fp_len
        self.mol_fp_radiis = mol_fp_radiis
        self.stem_fp_len = stem_fp_len
        self.stem_fp_radiis = stem_fp_radiis

    def __call__(self, molecule):
        mol = molecule.mol
        mol_fp = get_fp(mol, self.mol_fp_len, self.mol_fp_radiis)

        # get fingerprints and also handle empty case
        stem_fps = [get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx])
                    for idx in molecule.stem_atmidxs]

        jbond_fps = [(get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx[0]]) +
                     get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx[1]]))/2.
                     for idx in molecule.jbond_atmidxs]

        if len(stem_fps) > 0:
            stem_fps = np.stack(stem_fps, 0)
        else:
            stem_fps = np.empty(shape=[0, self.stem_fp_len * len(self.stem_fp_radiis)], dtype=np.float32)
        if len(jbond_fps) > 0:
            jbond_fps = np.stack(jbond_fps, 0)
        else:
            jbond_fps = np.empty(shape=[0, self.stem_fp_len * len(self.stem_fp_radiis)], dtype=np.float32)
        return mol_fp, stem_fps, jbond_fps


class MakeFP(object):
    """makes a fingerprint for molecule"""
    def __call__(self, data, fp_length=1024, radii=3):
        try:
            mol = Chem.MolFromSmiles(data.smi)
            fp = get_fp(mol, fp_length=1024, fp_radiis=[radii])
        except Exception as e:
            fp = np.zeros(fp_length, dtype=np.float32)

        data.fp = fp
        return data


# def get_fingerprint(smiles, radius=2,length=1024):
# fixme: make a part of get_fp
#
#   """Get Morgan Fingerprint of a specific SMILES string.
#   Args:
#     smiles: String. The SMILES string of the molecule.
#     hparams: tf.contrib.training.HParams. Hyper parameters.
#   Returns:
#     np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
#   """
#   if smiles is None:
#     return np.zeros((length,))
#   molecule = Chem.MolFromSmiles(smiles)
#   if molecule is None:
#     return np.zeros((length,))
#   fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, length)
#   arr = np.zeros((1,))
#   DataStructs.ConvertToNumpyArray(fingerprint, arr)
#   return pd.DataFrame({"fingerprint":[arr]})


def onehot(arr, num_classes, dtype=np.int):
    arr = np.asarray(arr, dtype=np.int)
    assert len(arr.shape) ==1, "dims other than 1 not implemented"
    onehot_arr = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    onehot_arr[np.arange(arr.shape[0]), arr] = 1
    return onehot_arr


_mpnn_feat_cache = [None]


def mpnn_feat(mol, ifcoord=True, panda_fmt=False, one_hot_atom=False, donor_features=True):
    atomtypes = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bondtypes = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    natm = len(mol.GetAtoms())
    ntypes = len(atomtypes)
    # featurize elements
    # columns are: ["type_idx" .. , "atomic_number", "acceptor", "donor",
    # "aromatic", "sp", "sp2", "sp3", "num_hs", [atomic_number_onehot] .. ])

    nfeat = ntypes + 1 + 8
    if one_hot_atom:
        nfeat += len(atomic_numbers)
    atmfeat = np.zeros((natm, nfeat))

    # featurize
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx = atomtypes.get(atom.GetSymbol(), 5)
        atmfeat[i, type_idx] = 1
        if one_hot_atom:
            atmfeat[i, ntypes + 9 + atom.GetAtomicNum() - 1] = 1
        else:
            atmfeat[i, ntypes + 1] = atom.GetAtomicNum()
        atmfeat[i, ntypes + 4] = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        atmfeat[i, ntypes + 5] = hybridization == HybridizationType.SP
        atmfeat[i, ntypes + 6] = hybridization == HybridizationType.SP2
        atmfeat[i, ntypes + 7] = hybridization == HybridizationType.SP3
        atmfeat[i, ntypes + 8] = atom.GetTotalNumHs(includeNeighbors=True)

    # get donors and acceptors
    if donor_features:
        if _mpnn_feat_cache[0] is None:
            fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            _mpnn_feat_cache[0] = factory
        else:
            factory = _mpnn_feat_cache[0]
        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
             if feats[j].GetFamily() == 'Donor':
                 node_list = feats[j].GetAtomIds()
                 for k in node_list:
                     atmfeat[k, ntypes + 3] = 1
             elif feats[j].GetFamily() == 'Acceptor':
                 node_list = feats[j].GetAtomIds()
                 for k in node_list:
                     atmfeat[k, ntypes + 2] = 1
    # get coord
    if ifcoord:
        coord = np.asarray([mol.GetConformer(0).GetAtomPosition(j) for j in range(natm)])
    else:
        coord = None
    # get bonds and bond features
    bond = np.asarray([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
    bondfeat = [bondtypes[bond.GetBondType()] for bond in mol.GetBonds()]
    bondfeat = onehot(bondfeat, num_classes=len(bondtypes))

    # convert atmfeat to pandas
    if panda_fmt:
        atmfeat_pd = pd.DataFrame(index=range(natm), columns=[
            "type_idx", "atomic_number", "acceptor", "donor", "aromatic",
            "sp", "sp2", "sp3", "num_hs"])
        atmfeat_pd['type_idx'] = atmfeat[:, :ntypes+1]
        atmfeat_pd['atomic_number'] = atmfeat[:, ntypes+1]
        atmfeat_pd['acceptor'] = atmfeat[:, ntypes+2]
        atmfeat_pd['donor'] = atmfeat[:, ntypes+3]
        atmfeat_pd['aromatic'] = atmfeat[:, ntypes+4]
        atmfeat_pd['sp'] = atmfeat[:, ntypes+5]
        atmfeat_pd['sp2'] = atmfeat[:, ntypes+6]
        atmfeat_pd['sp2'] = atmfeat[:, ntypes+7]
        atmfeat_pd['sp3'] = atmfeat[:, ntypes+8]
        atmfeat = atmfeat_pd
    return atmfeat, coord, bond, bondfeat


def mol_to_graph_backend(atmfeat, coord, bond, bondfeat, props={}, data_cls=Data):
    "convert to PyTorch geometric module"
    natm = atmfeat.shape[0]
    # transform to torch_geometric bond format; send edges both ways; sort bonds
    atmfeat = torch.tensor(atmfeat, dtype=torch.float32)
    if bond.shape[0] > 0:
        edge_index = torch.tensor(np.concatenate([bond.T, np.flipud(bond.T)], axis=1), dtype=torch.int64)
        edge_attr = torch.tensor(np.concatenate([bondfeat, bondfeat], axis=0), dtype=torch.float32)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, natm, natm)
    else:
        edge_index = torch.zeros((0, 2), dtype=torch.int64)
        edge_attr = torch.tensor(bondfeat, dtype=torch.float32)

    # make torch data
    if coord is not None:
        coord = torch.tensor(coord, dtype=torch.float32)
        data = data_cls(x=atmfeat, pos=coord, edge_index=edge_index, edge_attr=edge_attr, **props)
    else:
        data = data_cls(x=atmfeat, edge_index=edge_index, edge_attr=edge_attr, **props)
    return data


def mol_to_graph(smiles, num_conf=1, noh=True, feat="mpnn", dockscore=None, gridscore=None, klabel=None):
    " mol to graph convertor "
    mol, _, _ = build_mol(smiles, num_conf=num_conf, noh=noh)
    if feat == "mpnn":
        atmfeat, coord, bond, bondfeat = mpnn_feat(mol)
    else:
        raise NotImplementedError(feat)
    props = {}
    if dockscore is not None:
        props["dockscore"] = dockscore
    if gridscore is not None:
        props["gridscore"] = gridscore
    if klabel is not None:
        props["klabel"] = klabel
    graph = mol_to_graph_backend(atmfeat, coord, bond, bondfeat, props)
    return graph
