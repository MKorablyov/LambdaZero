# LambdaZero: search in the space of small molecules

## Install 
### On Beluga/ComputeCanada
The anaconda environment and datasets are already available in a shared folder

* Clone this repo
```bash
git clone https://github.com/MKorablyov/LambdaZero

```
* Setup Config
```bash
cd LamdaZero
cp ./misc/external_dirs_cfg external_dirs.cfg 
vi external_dirs.cfg # change the name of your Summaries folder
# LambdaZero needs the following folders: 
# (1) Programs:  Third-party softwares. These are already installed on beluga to /lustre03/project/6004852/mkkr/Programs
# (2) Datasets: Datasets not included in this repo. These are already installed on beluga to /lustre03/project/6004852/mkkr/Datasets
# (3) Summaries: where the program will write output. Please, indicate your own location for this
```

* Test if the setup is working on one of the most basic scripts, on an interactive session on Beluga
```bash
cd LambdaZero/examples/mpnn
salloc --time=1:0:0 --cpus-per-task=4 --gres=gpu:1 --mem=32G --account=rrg-bengioy-ad

# Load environment variables (this would load the python environment modules with gcc/cuda modules)
source LambdaZero/misc/beluga_load_env.sh

# add current working repo to python path to allow global imports IE: import LambdaZero.some_module
export PYTHONPATH="${PYTHONPATH}:/path-to-where-you-have-current-work-repo/LambdaZero" 

# run mpnn training script
python train_mpnn.py

# for batch submisisons check LambdaZero/misc
# and for the beluga documentation refer to docs.mila.quebec
```

### Install using Anaconda 
Installation is complicated in practice - better default option is to use shared env on Beluga
Install [anaconda](https://www.anaconda.com/products/individual)

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
Note that the `install-prog-data.sh` script should be executed from within the conda environment as some python
dependencies are required to download the external programs.

Finally, install LambdaZero with the following command:
```
pip install -e .
```


## Getting started
For running RL experiments, it is recommended to use 4 GPUs, 12 CPUs and 64GB memory. You can start an interactive session as follows:
```bash
salloc --time=3:0:0 --account=rrg-bengioy-ad --gres=gpu:4 -c 12 --mem=64G
``` 
Within the interactive session, you can load the environment with the following commands:
```bash
source /lustre03/project/6004852/mkkr/LambdaZero/misc/beluga_load_env.sh
export PYTHONPATH="${PYTHONPATH}:/path-to-where-you-have-current-work-repo/LambdaZero" 
```

Run PPO
```
cd ~/LambdaZero/LambdaZero/examples/PPO  
# To change the number of workers, GPUs used by the algorithm, refer to the config in `train_ppo.py`
python train_ppo.py ppo_graph_001
# you should see something like this

#+----------------------------------+----------+--------------------+-----------+------------------+------+--------+
#| Trial name                       | status   | loc                |    reward |   total time (s) |   ts |   iter |
#|----------------------------------+----------+--------------------+-----------+------------------+------+--------|
#| PPO_BlockMolGraphEnv_v3_4e681962 | RUNNING  | 192.168.2.216:4735 | -0.582411 |          27.1576 | 4000 |      1 |
#+----------------------------------+----------+--------------------+-----------+------------------+------+--------+
```
...
Run AlphaZero
```bash
cd ~/LambdaZero/LambdaZero/examples/AlphaZero
# To change the number of workers, GPUs used by the algorithm, refer to the config in `train_az.py`
python train_az.py az001
```
...
Run LambdaBO (RL+BO)
```bash
cd ~/LambdaZero/LambdaZero/examples/bayesian_models/rl
# To change the number of workers, GPUs used by the algorithm, refer to the config in `train_bayesian_reward.py`
python train_bayesian_reward.py
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

### Batch Jobs
A sample script for batch jobs is available in `misc\sample_job.sh`. Please set the appropriate path to your scripts as well as the resources you need. 

The job can be launched as follows:
```bash
sbatch sample_job.sh
```

## Addtional Useful Resources
[MILA cluster docs](https://docs.mila.quebec/). 
[MILA Compute Canada Docs](https://mila.docs.server.mila.quebec/cluster/compute-canada-cluster/index.html)
[Compute Canada Docs](https://docs.computecanada.ca/wiki/Compute_Canada_Documentation)

## Getting Involved
[Calendar](https://calendar.google.com/calendar?cid=bnNncTk1NjVobWozY3Z2czUyZHI5anNuZThAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ)  
[Slack](https://lambdazerogroupe.slack.com/)  
[Asana](https://app.asana.com/0/1176844015060872/list)  
