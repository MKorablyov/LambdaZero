# Load required modules
module load python/3.6 cuda/10.1 cudnn/7.6.5
module load nixpkgs/16.09  gcc/8.3.0 rdkit/2019.03.4
source /scratch/<username>/env/bin/activate

# Setup CUDA paths
export PATH=$CUDA_PATH/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_PATH/lib64
