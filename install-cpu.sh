#!/usr/bin/env bash
set -eu

. .parse-args.sh

torch_tensorflow="pytorch::cpuonly tensorflow"
CUDA=cpu

. .install.sh
