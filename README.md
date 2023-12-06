# Benchmarking DeepONet with JAX

This repo contains the following five sections:

1. [Installation](#1-installation): creating a fresh `conda` environment for the benchmark;

2. [Problem and data](#2-problem-and-data): problem description and data generation;

3. [Training and metrics](#3-training-and-metrics): training setups and metrics for comparison;

4. [Baseline](#4-baseline): one baseline implementation in pure `jax + flax` by Kuangdai;

5. [ZCS](#5-zcs): our ZCS implementation in pure `jax + flax` by Kuangdai.

# 1. Installation

The only required dependencies for this repo are `jax` and `flax`.
Optional, you can install `deepxde`, which is needed for data generation;
otherwise, you can choose to download pre-generated data, as detailed later.

The following lines provide an example FYR, starting from a new `conda` environment.

```bash
# env
conda create --name deeponet_jax_bench pip git tqdm DeepXDE
conda activate deeponet_jax_bench
# jax
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install flax
# deepxde (only needed for data generation)
pip install deepxde 
# clone repo
git clone https://github.com/kuangdai/deeponet-jax-bench
cd deeponet-jax-bench
```

**Using the same device and environment (such as CUDA version) is not too important for this benchmark,
as we mostly care about the relative (rather than absolute) GPU and time savings w.r.t. the baseline.**

# 2. Problem and data

The problem is adapted from one of the demonstrations of PI-DeepONets in `deepxde`,
[the diffusion reaction equation](https://github.com/lululxvi/deepxde/blob/master/examples/operator/diff_rec_aligned_pideeponet.py).
This is also an example in [DeepXDE-ZCS](https://github.com/stfc-sciml/DeepXDE-ZCS).
We use this example so that the results can be easily compared to those by the other backends
(`torch`, `tf` and `paddle`).

Use the following line to generate the full dataset:

```bash
DDE_BACKEND=jax python xgen_data.py
```

The following files will be generated in `data/`:

| **NAME**        | **TYPE** | **SIZE**        | **CONTAINS**                                                                                    | 
|-----------------|----------|-----------------|-------------------------------------------------------------------------------------------------|
| `xt_all.npy`    | `float`  | `(12000, 2)`    | Coordinates of collocation points on a 100 x 120 meshgrid with 100 $x$'s and 120 $t$'s          |
| `bc_idx.npy`    | `int`    | `(240,)`        | Indices of points on boundary condition ($x=0$ and $x=1$)                                       |
| `ic_idx.npy`    | `int`    | `(100,)`        | Indices of points on initial condition ($t=0$)                                                  |
| `features.npy`  | `float`  | `(1500, 50)`    | Input for the branch net, with 1500 functions and 50 features for each function                 |
| `sources.npy`   | `float`  | `(1500, 12000)` | Source fields in the PDE for the 1500 functions, which is time-independent but tiled along time |
| `solutions.npy` | `float`  | `(1500, 12000)` | Solution fields of the PDE for the 1500 functions                                               |

If you do not have `deepxde`, you
can [download the data (130 MB)](https://drive.google.com/file/d/1WBT7nXl21fdKxedo-fbWbiCo7MOLDeeI/view?usp=sharing)
instead.

# 3. Training and metrics

### Sampling a data batch

Because this benchmark has a focus on GPU memory and time measurements, we must make sure that
the neural network sees the same amount of data in one batch (i.e., for one time of model update).
**We set the number of functions at $M=50$ and the number of points at $N=4000$ for one batch**.
Data sampling is facilitated by the `DiffRecData` class under `src/`, e.g.,

```python
from src.utils import DiffRecData

data = DiffRecData(data_path='data/')
# if train==True, sample from the first 1000 functions; otherwise from the last 500
branch_input, trunk_input, source_input, u_true = data.sample_batch(
    n_funcs=50, n_points=4000, seed=0, train=True)
```

The outputs contain the data in a batch:

| **NAME**       | **SIZE**     | **CONTAINS**                                                                   | 
|----------------|--------------|--------------------------------------------------------------------------------|
| `branch_input` | `(50, 50)`   | Input for the branch net, with 50 functions and 50 features for each function  |
| `trunk_input`  | `(4340, 2)`  | Input for trunk net, with 4340 points (4340 = 4000 + 240 + 100)                |
| `source_input` | `(50, 4340)` | Source terms to be used for PDE calculation, with 50 functions and 4340 points |
| `u_true`       | `(50, 4340)` | Solutions to the PDE, with 50 functions and 4340 points                        |

**The solutions `u_true` can only be used for validation (i.e., not used in training).**

### Network architecture

The features for the branch and trunk nets are respectively `[50, 128, 128, 128]` and `[2, 128, 128, 128]`.
A `flax` implementation in the format of "cartesian product" is provided in `src/model.py`, which will be used for
our baseline and ZCS solutions. You do not need to use this implementation or the format of "cartesian product";
however, please make sure that **your network has the same amount of trainable parameters,
and it sees the same amount of data in each iteration as defined above**.

### Metrics

Report the following three numbers after **training for 1000 batches**:

* **Peak GPU memory**: we do not provide code to do this; you can monitor the memory using tools
  such as [nvitop](https://github.com/XuehaiPan/nvitop). Note that `jax` pre-allocates 75% of available GPU memory
  by default, and we must disable such behaviour by `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

* **Total wall time**: because `jax` can run so fast, please exclude the time used on data sampling
  (`DiffRecData.sample_batch`).

* **L2 relative error**: you can use `src.batched_l2_relative_error` to compute the error; after 1000 batches,
  it should come down to around 30%.

---

#### XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

The benchmark problem has been completely defined as above.

The rest is about our own baseline and ZCS solutions.

#### XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

---

# 4. Baseline

We provide a baseline solution:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python xtrain_baseline.py -M 50 -N 4000 
```

This solution adopts the format of "cartesian product", similar to
`PDEOperatorCartesianProd` in `deepxde`.
In `deepxde`, the function dimension is handled by a for-loop; here we use
`vmap` to vectorise this dimension.
The whole `train_step` is jit compiled.

Our measurements on a V100 GPU is reported as follows:

| **METHOD** | **GPU / MB** | **TIME / s** | $M=50, N=4000$ |
|------------|--------------|--------------|----------------|
| Baseline   | 2907         | 39           |                |

Compared to the original `deepxde` solutions with the other backends from
[DeepXDE-ZCS](https://github.com/stfc-sciml/DeepXDE-ZCS), one can see that this `jax` baseline
has surpassed all the other backends (`torch`, `tf` and `paddle`). 
However, note that the time measurements in `deepxde` include data sampling.

# 5. ZCS

Our ZCS solution can be obtained via

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python xtrain_zcs.py -M 50 -N 4000 
```

The measurements are as below. Similar to the other backends, the measurements show a significant reduction of
GPU memory consumption and wall time.

| **METHOD** | **GPU / MB** | **TIME / s** | $M=50, N=4000$ |
|------------|--------------|--------------|----------------|
| Baseline   | 2907         | 39           |                |
| ZCS        | 603          | 5            |                |

Further, we increase the problem scale by using `-M 100 -N 8000`, and the measurements are reported below:

| **METHOD** | **GPU / MB** | **TIME / s** | $M=100, N=8000$ |
|------------|--------------|--------------|-----------------|
| Baseline   | 10851        | 147          |                 |
| ZCS        | 867          | 8            |                 |

They show that ZCS gives more savings for larger-scale problems. 


