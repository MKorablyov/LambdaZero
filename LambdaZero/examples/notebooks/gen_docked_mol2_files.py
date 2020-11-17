import argparse
import pandas as pd
import ray

from pathlib import Path

from LambdaZero.chem import Dock_smi
from LambdaZero.utils import get_external_dirs

@ray.remote
def dock_ray(dock_smi, smiles, mol_name):
    _, _, _ = dock_smi.dock(smiles, mol_name=mol_name)


if __name__ == "__main__":
    """
    Produces .mol2 files with docked molecules (one per molecule) given .csv file that contains smiles strings (under the column named "smiles").

    Example: python3 gen_docked_mol2_files.py -i molecules.csv -o ./docked/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="path to the input .csv file that comprise smiles")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="path to the directory where to store output")
    parser.add_argument("-n", "--num_cpus", type=int, default=None, help="number of cpus to use (default: all)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    # create ouput directory if necessary
    output_dir = Path(args.output_dir) 
    output_dir.mkdir(parents=True, exist_ok=True)		 
    
    # set paths to binaries and docking parameters
    datasets_dir, programs_dir, _ = get_external_dirs()
    dock6_dir = Path(programs_dir, "dock6")
    chimera_dir = Path(programs_dir, "chimera")
    docksetup_dir = Path(datasets_dir, "brutal_dock/mpro_6lze/docksetup")
    
    # instantiate docking class
    dock_smi = Dock_smi(outpath=args.output_dir,
                    	chimera_dir=chimera_dir,
                    	dock6_dir=dock6_dir,
                    	docksetup_dir=docksetup_dir,
                    	gas_charge=True,
                    	trustme=True,
                    	cleanup=False)

    # perform docking
    ray.init(num_cpus=args.num_cpus)
    docking_tasks = [dock_ray.remote(dock_smi, row.smiles, str(index)) for index, row in df.iterrows()]
    ray.get(docking_tasks)
    ray.shutdown() 

    # erase files that we are not interested in
    files_to_remove = [filename for filename in output_dir.iterdir() if not filename.name.endswith('_scored.mol2')]    
    for filename in files_to_remove:
        filename.unlink()     

