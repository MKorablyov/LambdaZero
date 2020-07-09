#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --output=/home/sbasu/logs/out/generate_data_%j.txt
#SBATCH --account=def-bengioy
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00

# Sanity check
if [ "x$SLURM_TMPDIR" == "x" ]; then
    echo "Not running in SLURM context! Exiting!"
    exit
fi
# Module loads
module purge
module load python/3.6 scipy-stack/2019b cuda/10.1 cudnn/7.6.5 gcc/8.3.0
module load arrow/0.17.1
module load rdkit/2019.03.4
# Unpack the data
export WHEELS="$SLURM_TMPDIR"/wheels
echo "Unpacking wheel pack to $WHEELS ..."
unzip -d "$SLURM_TMPDIR" $HOME/scratch/lzwheels.zip
echo "Unpacked wheels at $WHEELS ."
# Make virtualenv
cd "$SLURM_TMPDIR"
virtualenv --no-download lz      # Make it
. lz/bin/activate                # Activate it
#
# Bypass broken CC 'install' command by installing a fixed version in the activated
# virtualenv's bin/, where it will be found with higher priority than the broken one.
#
echo '${NIXUSER_PROFILE}/bin/install --mode=u+rwx,go+rx,uo-s "$@"' > lz/bin/install
chmod +x lz/bin/install
#
# The lz4 3.1.0 python packages wants the lz4 library, but we don't have it.
# Build it from scratch, it being apparently missing.
#
tar -xf "$WHEELS"/lz4-1.9.2.tar.gz
cd lz4-1.9.2
make
PREFIX="$SLURM_TMPDIR"/lz make install
cd ..
rm -R lz4-1.9.2
#
# Prioritize installing the dependencies of the Python lz4 package.
# Then, provide the magic CFLAGS and LD_FLAGS so pip can install it.
#
pip install --no-index -f "$WHEELS" "$WHEELS"/pkgconfig-1.5.1-py2.py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/setuptools_scm-4.1.2-py2.py3-none-any.whl
CFLAGS="-I $SLURM_TMPDIR/lz/include" LD_FLAGS="-L $SLURM_TMPDIR/lz/lib" \
pip install --no-index -f "$WHEELS" "$WHEELS"/lz4-3.1.0.tar.gz
# CFLAGS="-I $SLURM_TMPDIR/lz/include" LD_FLAGS="-L $SLURM_TMPDIR/lz/lib" \
# pip install --no-index -f "$WHEELS" "$WHEELS"/lz4-3.1.0.tar.gz
#
# Export precious extra LD_LIBRARY_PATH variables related to lz4 and CUDA.
#
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$CUDA_PATH/lib64:$SLURM_TMPDIR/lz/lib"
# Pip install various other packages
pip install --no-index -f "$WHEELS" "$WHEELS"/pytest_runner-5.2-py2.py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/imagesize-1.2.0-py2.py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/multidict-4.7.6.tar.gz
pip install --no-index -f "$WHEELS" "$WHEELS"/yarl-1.4.2.tar.gz
pip install --no-index -f "$WHEELS" docutils==0.15.2 cloudpickle==1.3.0
pip install --no-index -f "$WHEELS" absl-py psutil plyfile pymongo isodate filelock rdflib pyyaml
pip install --no-index -f "$WHEELS" tqdm future colorama tabulate
pip install --no-index -f "$WHEELS" dm-tree click beautifulsoup4 Werkzeug Flask
pip install --no-index -f "$WHEELS" alabaster babel
pip install --no-index -f "$WHEELS" "$WHEELS"/idna-ssl-1.1.0.tar.gz
pip install --no-index -f "$WHEELS" "$WHEELS"/aiohttp-3.6.2-py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/Sphinx-3.1.1-py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/sphinx_rtd_theme-0.5.0-py2.py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/commonmark-0.9.1-py2.py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/recommonmark-0.6.0-py2.py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/typing_inspect-0.6.0-py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/typed-argument-parser-1.5.0.tar.gz
pip install --no-index -f "$WHEELS" "$WHEELS"/alembic-1.4.2.tar.gz
pip install --no-index -f "$WHEELS" "$WHEELS"/coverage-5.1.tar.gz
pip install --no-index -f "$WHEELS" "$WHEELS"/wandb-0.9.1-py2.py3-none-any.whl
pip install --no-index -f "$WHEELS" orion
pip install --no-index -f "$WHEELS" pytest boto3 scikit-learn scikit-image
pip install --no-index -f "$WHEELS" ase h5py==2.10.0 llvmlite==0.31.0 numba==0.48.0
pip install --no-index -f "$WHEELS" torch==1.5.1+cu101 torchvision==0.6.1+cu101
pip install --no-index -f "$WHEELS" torch_scatter==2.0.4+cu101 torch_sparse==0.6.5+cu101 \
                                    torch_cluster==1.5.5+cu101 torch_spline_conv==1.2.0+cu101 \
                                    torch_geometric==1.5.0
pip install --no-index -f "$WHEELS" tensorflow-gpu==2.2.0
pip install --no-index -f "$WHEELS" "$WHEELS"/fastparquet-0.4.0.tar.gz
pip install --no-index -f "$WHEELS" "$WHEELS"/msgpack-1.0.0.tar.gz
pip install --no-index -f "$WHEELS" "$WHEELS"/chemprop-0.0.3-py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/mlflow-1.9.0-py3-none-any.whl
pip install --no-index -f "$WHEELS" "$WHEELS"/ray-0.9.0.dev0-cp36-cp36m-linux_x86_64.whl[rllib]
# Execute your code
# exec "$@"

cd /home/sbasu/LambdaZero/
pip install -e .

python /home/sbasu/LambdaZero/LambdaZero/datasets/greedy_actor/greedy_actor_mpnn.py