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
To setup the environment for Beluga, create a virtual environment and install dependencies.
# Create virtual environments and install dependencies
```bash
module load python/3.6 cuda/10.1 cudnn/7.6.5
virtualenv --no-download ~/env
source ~/env/bin/activate
cd ~/LambdaZero
pip install -r requirements.txt
```

Add the following line to your slurm job file to load the required modules.
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
