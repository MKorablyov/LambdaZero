## Pretraining of the MPNN used as a reward model by LambdaZero

A sub task of LambdaZero is to _learn_ a reward function that bypass expensive Docking simulations or even more expensive 
Free Energy Perturbation simulations as accurately and cheaply as possible. Preparing such a model can be done separately 
as a supervised learning task. The goal is to train a model that will be plugged in LambdaZero's RL algorithm.

### Installation

LambdaZero must be installed using `install-cpu.sh` of `install-gpu.sh` from the root directory of the project. Then it might be necessary to install additional components required by `pandas` to read the dataset (which are not required for the default use of LambdaZero):

    conda install snappy
    pip install pyarrow  # conda install pyarrow failed for me

### Training the MPNN

#### example 1

    python brutal_dock_model.py

This will generate the graph dataset (expect about 1 or 2 hours on a good laptop), save the dataset for later use and train the MPNN (another hour or so without GPU). 

#### example 2

    python train_mpnn_example.py

This only generate a small dataset of graphs, load the pretrained model and train it. Because of the smaller dataset, the results are not to be trusted. This script serves as _the simplest_ example of the training pipeline. It was initially intended to find the necessary parts of LambdaZero for the representation learning sub-project.

### Where is the data?

The dataset raw file (for D4, not for covid) contains smiles strings (our input) and docking scores (our targets). These are saved as `pandas` dataframe located at

    .../LambdaZero/Datasets/brutal_dock/d4/raw/dock_blocks105_walk40_clust.feather
    .../LambdaZero/Datasets/brutal_dock/d4/raw/dock_blocks105_walk40_2_clust.feather

The dataframe columns are `'simles'` and `'gridscore'`. The column `'klabel'` is also used by `brutal_dock_model.py`, but I donn't know its purpose at the time of this writing. The targets (`gridscore`) must be normalized for training `(y-y.mean())/y.std()` to be comparable with the original model.

`brutal_dock_model.py` first builds the molecular graphs for each SMILES string and put those in a custom dataset object (grandchildren of `torch-geometric.Dataset`). The latter is saved for later use in

    .../LambdaZero/Datasets/brutal_dock/d4/processed/dock_blocks105_walk40_clust.pt
    .../LambdaZero/Datasets/brutal_dock/d4/processed/dock_blocks105_walk40_2_clust.pt

### How does LambdaZero use the model?

Here are some hints about how the MPNN is used in LambdaZero (as of #533a619).

The class `Net`, implemented and used in the training script `brutal_dock_model.py` is not the one used by the actual reinforcement learning algorithm. The latter use an identical copy, named `MPNNet` implemented in `LambdaZero/environments/molecule.py`.

This `MPNNet` is only instanciated in the class `PredDockReward`.

The only instanciation of `PredDockReward` happens in `BlockMolEnv_v3`. The latter is _very_ implicit:

    self.reward = config['reward'](**config["reward_config"])
    
Here, `config['reward']` correspond to `PredDockReward`,  and `config["reward_config"]` are the arguments passed to the instanciation. To see this, you must follow, in the code, the chain of definitions leading to the variable `congig`. 

The `PredDockReward` is ultimately called when the RL algorithm is ran. One such call happens in `test_env.py`:

    obs, reward, don, info = env.step(action)

where `env` is an intance of `BlockMolEnv_v3`. The methods `BlockMolEnv_v3.step` and `PredDockReward._simulation` giveaway the necessary data preparation required by the model.
