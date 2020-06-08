import pytest

from LambdaZero.chem.chimera_op import get_chimera_commands


@pytest.fixture
def input_mol_file_path():
    return 'test_input_file_path'


@pytest.fixture
def charge_method():
    return 'test_charge_method'


@pytest.fixture
def output_mol2_file_path():
    return 'test_output_file_path'


@pytest.fixture
def expected_command(input_mol_file_path, charge_method, output_mol2_file_path):
    chimera_cmd = f"""open {input_mol_file_path}
addh
addcharge all method {charge_method}
write format mol2 0 {output_mol2_file_path}
stop now"""
    return chimera_cmd


def test_get_chimera_command(input_mol_file_path, charge_method, output_mol2_file_path, expected_command):

    computed_command = get_chimera_commands(input_mol_file_path, charge_method, output_mol2_file_path)

    assert computed_command == expected_command
