# LambdaZero: search in the space of small molecules

## Install
Install anaconda: [miniconda](https://docs.conda.io/en/latest/miniconda.html)

Create the conda environment, based on your os:
```
conda env create -f ( environment-linux.yml |environment-mac.yml ) [-n env_name]
```
This will create an environment named `lz` by default. Activate this environment
```
conda activate [env_name]
```

LambdaZero depends on external programs (such as Dock6 and UCSF Chimera) and datasets (brutal dock and fragdb etc. ) that are not provided in this repo. These can be installed by running:
```
bash install-prog-datasets.sh [-d dataset_path] [-p programs_path] [-s summaries_path]
```
this script would create a locator file called `external_dirs.cfg` that is machine specific and is used by the LambdaZero core to be able to call external dependencies. 
Note that the `install-prog-datasets.sh` script should be executed from within the conda environment as some python
dependencies are required to download the external programs.

Finally, install LambdaZero with the following command:
```
pip install -e .
```

### Setup on Beluga
To setup the environment for Beluga, create a virtual environment and install dependencies. Replace `<username>` with your username.
# Create virtual environments and install dependencies
```bash
module load python/3.6 cuda/10.1 cudnn/7.6.5
virtualenv --no-download /scratch/<username>/env
source /scratch/<username>/env/bin/activate
# torch and tensorflow need be installed separately
pip install numpy
pip install tensorflow_gpu --no-index
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-cluster==1.5.5+cu101 torch-scatter==2.0.4+cu101 torch-sparse==0.6.5+cu101 torch-spline-conv==1.2.0+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
# install ray
wget -O ray-0.9.0.dev0-cp36-cp36m-linux_x86_64.whl https://ray-wheels.s3-us-west-2.amazonaws.com/master/11c2b500b86e1a02a97f65708796e1605eb91808/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl
pip install ray-0.9.0.dev0-cp36-cp36m-linux_x86_64.whl
cd ~/LambdaZero
pip install -r requirements.txt
# install LambdaZero
pip install -e .
```

Add the following line to your slurm job file to load the required modules. Replace `<username>` with your username.
```bash
bash setup_beluga.sh
```

This will load the required modules and install the dependencies.

## Getting started

Run PPO
```
cd ~/LambdaZero/LambdaZero/examples/PPO  
python train_ppo.py ppo001
# you should see something like this

#+-----------------------------+----------+--------------------+-----------+------------------+------+--------+
#| Trial name                  | status   | loc                |    reward |   total time (s) |   ts |   iter |
#|-----------------------------+----------+--------------------+-----------+------------------+------+--------|
#| PPO_BlockMolEnv_v3_4e681962 | RUNNING  | 192.168.2.216:4735 | -0.582411 |          27.1576 | 4000 |      1 |
#+-----------------------------+----------+--------------------+-----------+------------------+------+--------+

...
```
Run Ape-X
```
cd ~/LambdaZero/LambdaZero/examples/PPO  
python train_apex.py apex001
```
Run AlphaZero
```
cd ~/LambdaZero/LambdaZero/examples/AlphaZero
# az000 ending by three zeros means it is a debug configuration in this case it means expanding MCTS only a few times instead of 800 or 1600 times as in the original implementation to make sure the algorithm runs.
python train_az.py az000
```
Train vanilla MPNN on biophysics simulation data
```
cd ~/LambdaZero/LambdaZero/datasets/brutal_dock 
python split_random.py
cd ~/LambdaZero/LambdaZero/examples/mpnn
python train_mpnn.py
```

Use environment, make random walks, call oracles:

```
cd ~/LambdaZero/LambdaZero/examples/oracles
python oracle_examples.py
```

## Getting Involved
[Calendar](https://calendar.google.com/calendar?cid=bnNncTk1NjVobWozY3Z2czUyZHI5anNuZThAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ)  
[Slack](https://lambdazerogroupe.slack.com/)  
[Asana](https://app.asana.com/0/1176844015060872/list)  
