#!/usr/bin/env bash
set -eu

. .parse-args.sh

torch_tensorflow="cudatoolkit=10.1 tensorflow-gpu"
CUDA=cu101

. .install.sh
