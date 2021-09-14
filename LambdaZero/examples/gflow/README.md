### 
### Usage 

Req: `liftoff` (`pip install git+git://github.com/tudor-berariu/liftoff.git#egg=liftoff`)

**Run 1 config file**: 

`liftoff LambdaZero.examples.gflow.train.py LambdaZero/examples/gflow/configs/base.yaml`

**Run batch config file**: 

_runs-no - seeds /experiment_

1. This will generate a batch of configs in a new folder  (let's call this new folder `exp_batch`, it will be printed at the console).

`liftoff-prepare LambdaZero/examples/gflow/configs/demogrid/ --runs-no 3 --results-path /scratch/andrein/gflow/results  --do`


2. We use the directory path produced by the above step and we can run `liftoff` on that directory to run all configs one by one (or just 1 of the configs using `--max-runs 1`):

`liftoff LambdaZero/examples/gflow/train.py --max-runs 1 --no-detach ${exp_batch}`

This way we can easily run a batch of experiments on the cluster, see `LambdaZero/examples/gflow/example_sbatch_array.sh`.

We can clean failed experiments in a folder using e.g.:

`liftoff-clean --clean-all --crashed-only /scratch/andrein/gflow/results/exp_batch/ --do`


**Proxy info, preloaded model scores**:

500 Test set (top 500 candidates from `dock_db_1624547349tp_2021_06_24_11h.h5`)

```
MAE Mean : 2.4388244991302486 | max: 5.020869636535645 | min : 1.2189960479736328
```

Test set info:
```
count    500.000000
mean     -14.737000
std        0.235206
min      -15.900000
25%      -14.800000
50%      -14.700000
75%      -14.600000
max      -14.500000
Name: dockscore, dtype: float64
```

```
count    500.000000
mean       0.332065
std        0.031444
min        0.300017
25%        0.310547
50%        0.327631
75%        0.345987
max        0.540806
Name: qed_score, dtype: float64
```

```
count    500.000000
mean       5.614035
std        0.887053
min        4.006549
25%        4.929133
50%        5.673222
75%        6.225841
max        7.983991
Name: synth_score, dtype: float64
```
