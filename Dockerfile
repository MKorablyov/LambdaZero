FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ARG CUDA=cpu

# Install all the conda-available packages
RUN conda install -y \
        tensorflow \
        pandas \
        conda-forge::rdkit \
        networkx \
        scikit-image \
        scikit-learn \
        numba \
        isodate \
        jsonschema \
        redis-py \
        pyyaml \
        colorama \
        filelock \
        aiohttp \
        beautifulsoup4 \
        future \
        lz4 \
        tabulate \
        fastparquet \
        boto3

# Install pytorch-geometric special dependencies
RUN pip install --no-cache-dir \
        torch-scatter==latest+${CUDA} \
        torch-sparse==latest+${CUDA} \
        torch-cluster==latest+${CUDA} \
        torch-spline-conv==latest+${CUDA} \
        -f https://pytorch-geometric.com/whl/torch-1.4.0.html

# Need to install this first
RUN pip install --no-cache-dir \
        https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp37-cp37m-manylinux1_x86_64.whl \
    # Install the pip packages
    && pip install --no-cache-dir \
        psutil \
        torch-geometric \
        ray[rllib]

COPY . LambdaZero

RUN pip install --no-cache-dir -e ./LambdaZero \
    && mkdir -p Datasets \
    && cd Datasets \
    && git clone --depth 1 https://github.com/MKorablyov/fragdb \
    && git clone --depth 1 https://github.com/MKorablyov/brutal_dock \
    && cd .. \
    && mkdir -p Programs \
    && cd Programs \
    && git clone --depth 1 https://github.com/MKorablyov/dock6 \
    && git clone --depth 1 https://github.com/MKorablyov/chimera tmp \
    && cd tmp \
    && cat xaa xab > chimera.bin \
    && chmod 755 chimera.bin \
    && echo '../chimera' | ./chimera.bin \
    && cd .. \
    && rm -rf tmp
