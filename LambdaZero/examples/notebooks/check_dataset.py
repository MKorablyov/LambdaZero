import os.path as osp
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from torch_geometric import transforms as T
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
import LambdaZero.utils
import LambdaZero.inputs


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
transform = T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()])

config = {"dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore", "smi"],
            "transform": transform,
            "file_names":
            #    ["Zinc15_2k"],
            ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],

        },
          "dataset_split_path": osp.join(datasets_dir,
                                         #"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
          "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
          "target":"gridscore",


          }




# dataset = config["dataset"](**config["dataset_config"])
# targets = np.asarray([getattr(d, config["target"]).cpu().numpy() for d in dataset])
# print("min train targets", targets.min(), targets.std())
# sns.distplot(targets)

train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)

# train_set = Subset(dataset, train_idxs.tolist())
# val_set = Subset(dataset, val_idxs.tolist())
# test_set = Subset(dataset, test_idxs)
# train_loader = DataLoader(train_set, shuffle=True, batch_size=10)
# val_loader = DataLoader(val_set, batch_size=10)
# test_loader = DataLoader(test_set, batch_size=10)
#
# targets = np.asarray([getattr(d, config["target"]).cpu().numpy() for d in train_loader.dataset])
# sns.distplot(targets)
# targets = np.asarray([getattr(d, config["target"]).cpu().numpy() for d in val_loader.dataset])
# sns.distplot(targets)
# targets = np.asarray([getattr(d, config["target"]).cpu().numpy() for d in test_loader.dataset])
# sns.distplot(targets)

print(np.unique(train_idxs).shape[0] + np.unique(val_idxs).shape[0] + np.unique(test_idxs).shape[0])
print(np.unique(np.concatenate([train_idxs, test_idxs, val_idxs])).shape)




#config["dataset_config"]["file_names"] = ["Zinc15_2k"]
#dataset = config["dataset"](**config["dataset_config"])
#targets = np.asarray([getattr(d, config["target"]).cpu().numpy() for d in dataset])
#sns.distplot(targets)
plt.show()

