import subprocess


def get_chimera_commands(input_mol_file_path: str, charge_method: str, output_mol2_file_path: str):
    chimera_cmd = f"open {input_mol_file_path}\n" + \
                  "addh\n" + \
                  f"addcharge all method {charge_method}\n" + \
                  f"write format mol2 0 {output_mol2_file_path}\n" + \
                  "stop now"

    return chimera_cmd


def add_hydrogens_and_compute_gasteiger_charges_with_chimera(input_mol_file_path: str, charge_method: str,
                                                             chimera_bin: str, output_mol2_file_path: str):

    chimera_commands = get_chimera_commands(input_mol_file_path, charge_method, output_mol2_file_path)
    bash_command = f"printf \"{chimera_commands}\" | {chimera_bin} --nogui"
    process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE)
    process.wait()
