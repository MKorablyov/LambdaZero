import os
import numpy as np
import LambdaZero.utils
import LambdaZero.chem
from matplotlib import pyplot as plt
import seaborn as sns

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()

dock_path = os.path.join(datasets_dir, "temp_docking")
dock = LambdaZero.chem.DockVina_smi(dock_path)

try:
    score = dock.dock("O=C(NC12CC3CC(CC(C3)C1)C2)c1cc2nc(-c3ccco3)cc(C(F)(F)F)n2n1", mol_name="16")[1][0]
    # O=C1C=CC(=O)C(c2cccc3cc(C(=O)NC45CC6CC(CC(C6)C4)C5)ccc23)=C1
    # "O=C(NC(=O)C1=CC(C2CCCC2)CCC1)Nc1ccc2ccccc2c1")
except Exception as e:
    print(e)

# np.save(os.path.join(cfg.dock_path, "scores.npy"), np.asarray(scores))
# scores_local = (-np.load(os.path.join(cfg.dock_path, "scores.npy")) - 8.6) / 1.1
# scores_beluga = (-np.load(os.path.join(cfg.dock_path, "beluga_scores.npy")) - 8.6) / 1.1
# sns.distplot(scores_local)
# sns.distplot(scores_beluga)
# plt.show()