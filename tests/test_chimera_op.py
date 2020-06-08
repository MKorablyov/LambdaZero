import logging
import tempfile
from pathlib import Path

import pytest
from rdkit import Chem

from LambdaZero.chem.chimera_op import get_chimera_commands, add_hydrogens_and_compute_gasteiger_charges_with_chimera
from LambdaZero.utils import get_external_dirs


@pytest.fixture
def mol(easy_smiles):
    smiles = easy_smiles[0]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    return mol


@pytest.fixture
def input_mol_file_path(mol):

    mol_blocks = Chem.MolToMolBlock(mol)
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        mol_file_path = str(Path(tmp_dir_str).joinpath('test_input_mol_file.mol'))
        with open(mol_file_path, 'w+') as f:
            f.write(mol_blocks)
        yield mol_file_path
    logging.info("deleting test folder")


@pytest.fixture
def charge_method():
    return "gas"


@pytest.fixture
def output_mol2_file_path():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        mol2_file_path = str(Path(tmp_dir_str).joinpath('test_output_mol2_file.mol2'))
        yield mol2_file_path
    logging.info("deleting test folder")

    return 'test_output_file_path'


@pytest.fixture
def expected_command(input_mol_file_path, charge_method, output_mol2_file_path):
    chimera_cmd = f"""open {input_mol_file_path}
addh
addcharge all method {charge_method}
write format mol2 0 {output_mol2_file_path}
stop now"""
    return chimera_cmd


@pytest.fixture
def chimera_bin():
    _, programs_dir, _ = get_external_dirs()
    return str(Path(programs_dir).joinpath("chimera/bin/chimera"))


def test_get_chimera_command(input_mol_file_path, charge_method, output_mol2_file_path, expected_command):

    computed_command = get_chimera_commands(input_mol_file_path, charge_method, output_mol2_file_path)

    assert computed_command == expected_command


@pytest.mark.external_program
def test_add_hydrogens_and_compute_gasteiger_charges_with_chimera(input_mol_file_path, charge_method,
                                                                  chimera_bin, output_mol2_file_path):
    """
    In order to run this test, pytest must be invoked with the command line argument:
        pytest --external_program
    Otherwise this test will be skipped, as the chimera program is not always installed.
    """
    add_hydrogens_and_compute_gasteiger_charges_with_chimera(input_mol_file_path, charge_method,
                                                             chimera_bin, output_mol2_file_path)

    assert Path(output_mol2_file_path).is_file()


