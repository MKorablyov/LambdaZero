#!/usr/bin/env bash
set -eu

if [ -z "$1" ]; then
    echo "Pass the conda env name as argument: "
    echo ""
    echo "    ./install.sh lz"
    echo ""
    exit 1
fi

conda create -n "$1" -y python=3.6
CONDA_BIN=`which conda`
TMP=`dirname "$CONDA_BIN"`
CONDA_DIR=`dirname "$TMP"`

# Source the conda shell functions
. "$CONDA_DIR/etc/profile.d/conda.sh"

conda activate "$1"

# Install all the conda-available packages
conda install -y pytorch::pytorch==1.4 pytorch::torchvision cudatoolkit=10.1 tensorflow-gpu conda-forge::rdkit
#pandas networkx scikit-image scikit-learn numba isodate jsonschema redis-py pyyaml colorama filelock aiohttp beautifulsoup4 future lz4 tabulate fastparquet

# Install pytorch-geometric special dependencies
CUDA=cu101
pip install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} torch-cluster==latest+${CUDA} torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

# Need to install this first
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl

# Install the pip packages
pip install psutil torch-geometric ray[rllib] boto3 fastparquet

pip install -e .

mkdir -p Datasets
cd Datasets
git clone --depth 1 https://github.com/MKorablyov/fragdb
git clone --depth 1 https://github.com/MKorablyov/brutal_dock
cd ..

mkdir -p Programs
cd Programs
git clone --depth 1 https://github.com/MKorablyov/dock6
git clone --depth 1 https://github.com/MKorablyov/chimera tmp
cd tmp
cat xaa xab > chimera.bin
chmod 755 chimera.bin
echo '../chimera' | ./chimera.bin
cd ..
rm -rf tmp
cd ..

echo "Use \"conda activate $1\" to activate the conda environement."
