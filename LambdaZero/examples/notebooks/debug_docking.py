import os
import numpy as np
import LambdaZero.utils
import LambdaZero.chem

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

class cfg:
    num_sim = 2
    dock_path = os.path.join(datasets_dir,"temp_docking")

dock = LambdaZero.chem.DockVina_smi(cfg.dock_path)


scores = []
for i in range(cfg.num_sim):
    try:
        _, score, _ = dock.dock("O=C(NC(=O)C1=CC(C2CCCC2)CCC1)Nc1ccc2ccccc2c1")
        scores.append(score)
    except Exception as e:
        print(e)

np.save(os.path.join(cfg.dock_path, "scores.npy"), np.asarray(scores))
print(np.load(os.path.join(cfg.dock_path, "scores.npy")))