import os.path as osp
import os
import numpy as np
from ray.tune import grid_search
import LambdaZero.utils
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mcdrop, train_mpnn_brr, \
    mpnn_brr_mean_variance
    
datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()


mcdrop000 = {}

# default bayesian ridge regression 001-002
mcdrop001 = {"regressor_config":
                 {"config":
                      {"train":train_mpnn_brr,
                       "get_mean_variance": mpnn_brr_mean_variance,
                       }}}



mcdrop002 = {"regressor_config":
                 {"config":
                      {"train":train_mpnn_brr,
                       "get_mean_variance": mpnn_brr_mean_variance,
                        "dataset_split_path":
                           osp.join(datasets_dir,"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
                        "dataset_config": {
                          "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]}
                       }}}
# adjust lambda; and figure out where do dropout on large/small dataset
mcdrop003= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-4,-12,num=10))),
                       }}}
mcdrop004= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-4,-12,num=10))),
                       "model_config": {"drop_data":False, "drop_weights":True, "drop_last":True}
                       }}}
mcdrop005= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-4,-12,num=10))),
                       "model_config": {"drop_data":True, "drop_weights":True, "drop_last":True}
                       }}}

mcdrop006= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-4,-12,num=10))),
                       "dataset_split_path":
                           osp.join(datasets_dir,"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
                       "dataset_config": {
                           "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]}
                      }}}

mcdrop007= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-4,-12,num=10))),
                       "model_config": {"drop_last":True, "drop_data":True, "drop_weights":True},
                       "dataset_split_path":
                           osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
                       "dataset_config": {
                           "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]}
                      }}}


mcdrop_rl = {
     "regressor_config": {
          "config": {
               "lambda": 5.99e-11,
               "dataset_split_path":
                    osp.join(datasets_dir,"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
               "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]}
          }
     }
}







import torch
from LambdaZero.examples.bayesian_models.bayes_tune.functions import get_tau, train_epoch_with_targets, eval_epoch, \
    train_mcdrop_rl, mcdrop_mean_variance

mcdrop_debug = {
    'regressor_config': {
        "config": {
          "lambda": 6.16e-9,
          "T": 20,
          "lengthscale": 1e-2,
          "uncertainty_eval_freq":15,
          "train_epochs": 72,
          "finetune_epochs": 16,
          "model": LambdaZero.models.MPNNetDrop,
          "model_config": {"drop_data":False, "drop_weights":False, "drop_last":True, "drop_prob":0.1},
          "optimizer": torch.optim.Adam,
          "optimizer_config": {
              "lr": 0.001
          },
          "train_epoch": train_epoch_with_targets,
          "eval_epoch": eval_epoch,
          "train": train_mcdrop_rl,
          "get_mean_variance": mcdrop_mean_variance,
          "is_reward_model": True
        }
    },
}