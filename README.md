# LambdaZero: search in the space of small molecules

## Install 
### On Beluga/ComputeCanada
The anaconda environment and datasets are already available in a shared folder

```
# Clone this repo
git clone https://github.com/MKorablyov/LambdaZero

# LambdaZero needs the following folders: 
# (1) Third-party softwares. These are already installed on beluga to /lustre03/project/6004852/mkkr/Programs
# (2) Datasets not included in this repo. These are already installed on beluga to /lustre03/project/6004852/mkkr/Datasets
# (3) Summaries where the program will write output. Please, indicate your own location for this
cd LamdaZero
cp ./misc/external_dirs.cfg . 
vi external_dirs.cfg # change the name of your Summaries folder

# Test if the setup is working on one of the most basic scripts
cd ../LambdaZero/examples/mpnn
salloc --time=1:0:0 --cpus-per-task=4 --gres=gpu:1 --mem=32G --account=rrg-bengioy-ad

# Load environment variables (this would load the python environment modules with everything installed
bash /lustre03/project/6004852/mkkr/LambdaZero/misc/beluga_load_env.sh

# run mpnn training script
python train_mpnn.py

# for batch submisisons check LambdaZero/misc
# and for the beluga documentation refer to docs.mila.quebec

```



### Install using Anaconda 
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
Note that the `install-prog-datasets.sh` script should be executed from within the conda environment as some python
dependencies are required to download the external programs.

Finally, install LambdaZero with the following command:
```
pip install -e .
```


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

## Useful Resources
MILA cluster docs(https://docs.mila.quebec/). 

## Getting Involved
[Calendar](https://calendar.google.com/calendar?cid=bnNncTk1NjVobWozY3Z2czUyZHI5anNuZThAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ)  
[Slack](https://lambdazerogroupe.slack.com/)  
[Asana](https://app.asana.com/0/1176844015060872/list)  
