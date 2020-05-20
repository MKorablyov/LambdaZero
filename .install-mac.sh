### TODO: This file is to compile on a mac: it shouldn't exist, the main file should detect the OS.

conda create -n "$CONDA_ENV_NAME" -y python=3.6

# extract the base conda directory. This can be tricky if the executable is a symlink.
# The commands below performs the following steps:
#   1 - extract the string that contains the root prefix; the output is of the form "root_prefix: [conda base dir]"
#   2 - use awk to to extract [conda base dir]
#   3 - use sed to remove any blank spaces
CONDA_DIR=`conda config --show root_prefix | awk '{split($0, array, ":"); print array[2]}' | sed 's/ //g'`


set +u
# conda uses unboud variables
# Source the conda shell functions
. "$CONDA_DIR/etc/profile.d/conda.sh"

conda activate "$CONDA_ENV_NAME"
set -u

# Install all the conda-available packages
conda install -y $torch_tensorflow pytorch::pytorch==1.4.0 pytorch::torchvision conda-forge::rdkit pandas networkx scikit-image scikit-learn numba isodate jsonschema redis-py pyyaml colorama filelock aiohttp beautifulsoup4 future lz4 tabulate fastparquet boto3 pytest pytest-cov pyarrow ipython mlflow tqdm
 
# Prevent conda from being stupid about versions
conda config --env --add pinned_packages pytorch==1.4

# Install pytorch-geometric special dependencies
pip install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} torch-cluster==latest+${CUDA} torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

# Need to install this first
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp36-cp36m-macosx_10_13_intel.whl



# Install the pip packages
pip install orion wandb psutil torch-geometric ray[rllib]

# Install Chemprop
pip install git+https://github.com/chemprop/chemprop.git#egg=chemprop
pip install typed-argument-parser


pip install -e .


if [ $DATASET_DEFAULT -eq 1 ]; then
    read -p "Install directory for datasets (NO to skip) [$DATASET_PATH]:" dpath
    if [ ! -z "$dpath" ]; then
        DATASET_PATH="$dpath"
    fi
fi

if [ "$DATASET_PATH" != "NO" ]; then
    mkdir -p "$DATASET_PATH"
    cd $DATASET_PATH
    git clone --depth 1 https://github.com/MKorablyov/fragdb
    git clone --depth 1 https://github.com/MKorablyov/brutal_dock
    cd ..
fi


if [ $PROGRAMS_DEFAULT -eq 1 ]; then
    read -p "Install directory for programs (NO to skip) [$PROGRAMS_PATH]:" ppath
    if [ ! -z "$ppath" ]; then
        PROGRAMS_PATH="$ppath"
    fi
fi

if [ "$PROGRAMS_PATH" != "NO" ]; then
    mkdir -p "$PROGRAMS_PATH"
    cd $PROGRAMS_PATH
    git clone --depth 1 https://github.com/MKorablyov/dock6
    git clone --depth 1 https://github.com/MKorablyov/chimera tmp
    cd tmp
    cat xaa xab > chimera.bin
    chmod 755 chimera.bin
    echo '../chimera' | ./chimera.bin
    cd ..
    rm -rf tmp
    cd ..
fi

echo "Use \"conda activate $CONDA_ENV_NAME\" to activate the conda environement."
