![alt tag](doc/molMCTS.png)

# LambdaZero is a universal method for molecule optimization (in progress)
Library Includes:
- RL Algorithms
- Tools for dataset generation
- Tools to design action space

## Install
```
# the installation process is a bit tricky because of the rdkit package which only works properly with anaconda env
# create a new conda environment
conda create -n lz python=3.6
conda activate lz

# install ray nightly and ray[rllib]
pip install -U ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl
pip install ray[rllib]

# install pytorch and tensorflow
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install tensorflow

# install torch geometric package for graph processing
install torch geometric https://github.com/rusty1s/pytorch_geometric

# install rdkit
conda install -c conda-forge rdkit

# install misc
conda install -c anaconda pandas
pip install psutil

# install datasets
cd ~/Datasets
git clone https://github.com/MKorablyov/fragdb 	        # fragments to use in the scoring function
git clone https://github.com/MKorablyov/brutal_dock     # pretrained message passing NN to predict docking energy

# install 3rd party binaries
cd ~/Programs
# chimera is used for conversion between molecule formats and for visualization
# https://www.cgl.ucsf.edu/chimera/
git clone https://github.com/MKorablyov/chimera
# dock6 is a software the predicts energy of binding of the drug and protein
# http://dock.compbio.ucsf.edu/DOCK_6/index.html
git clone https://github.com/MKorablyov/dock6

# perform postclone instructions described at git clone https://github.com/MKorablyov/chimera

```

## Getting started
Run a few RL algorithms
`cd examples`
run ppo
```
python train_molecule.py ppo001
# you should see something like this

#+-----------------------------+----------+--------------------+-----------+------------------+------+--------+
#| Trial name                  | status   | loc                |    reward |   total time (s) |   ts |   iter |
#|-----------------------------+----------+--------------------+-----------+------------------+------+--------|
#| PPO_BlockMolEnv_v3_4e681962 | RUNNING  | 192.168.2.216:4735 | -0.582411 |          27.1576 | 4000 |      1 |
#+-----------------------------+----------+--------------------+-----------+------------------+------+--------+
#...
#+-----------------------------+----------+--------------------+-----------+------------------+-------+--------+
#| Trial name                  | status   | loc                |    reward |   total time (s) |    ts |   iter |
#|-----------------------------+----------+--------------------+-----------+------------------+-------+--------|
#| PPO_BlockMolEnv_v3_4e681962 | RUNNING  | 192.168.2.216:4735 | 0.0797528 |          103.641 | 20000 |      5 |
#+-----------------------------+----------+--------------------+-----------+------------------+-------+--------+
# ...
# +-----------------------------+----------+--------------------+----------+------------------+-------+--------+
#| Trial name                  | status   | loc                |   reward |   total time (s) |    ts |   iter |
#|-----------------------------+----------+--------------------+----------+------------------+-------+--------|
#| PPO_BlockMolEnv_v3_4e681962 | RUNNING  | 192.168.2.216:4735 | 0.501229 |          198.321 | 40000 |     10 |
#+-----------------------------+----------+--------------------+----------+------------------+-------+--------+

...


```

Run Ape-X
```
python train_molecule.py apex001
```

Run AlphaZero
```
# az000 ending by three zeros means it is a debug configuration
# in case of AlphaZero, it means expanding MCTS only a few times to see the outpus
python train_molecule.py az000
```

Run Docking simulation in parallel

Train MPNN on simulation data

For more information see tutorials, and docs

## Getting Involved
 Google group
 Slack community
 Meetings schedule


## References
- AlphaZero: https://arxiv.org/abs/1712.01815
- Ranked rewards: https://arxiv.org/abs/1807.01672


