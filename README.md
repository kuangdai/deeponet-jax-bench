# Benchmarking DeepONet with JAX

This repo contains the following five sections:

1. [Installation](#1-installation): creating a fresh `conda` environment for the benchmark;

2. [Problem and data](#2-problem-and-data): problem description and data generation;

3. [Training and metrics](#3-training-and-metrics): training setups and metrics for comparison;

4. [Baseline](#4-baseline): one baseline implementation in pure `jax + flax` by Kuangdai;

5. [Contributed solutions](#5-contributed-solutions): contributed solutions.

# 1. Installation

The only required dependencies for this repo are `jax` and `flax`.
Optionally, you can install `deepxde`, which is needed only for data generation;
otherwise, you can choose to download pre-generated data, as detailed later.

The following lines provide an example FYR, starting from a new `conda` environment.

```bash
# env
conda create --name deeponet_jax_bench pip git tqdm
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

Using the same device and environment (such as CUDA version) is not too important for this benchmark,
as we only care about the **relative (rather than absolute)** GPU and time savings w.r.t. the baseline.

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

| **NAME**        | **TYPE** | **SIZE**        | **CONTAINS**                                                                                                  | 
|-----------------|----------|-----------------|---------------------------------------------------------------------------------------------------------------|
| `xt_all.npy`    | `float`  | `(12000, 2)`    | Coordinates of collocation points on a $100 \times 120$ meshgrid with 100 $x$'s and 120 $t$'s                 |
| `bc_idx.npy`    | `int`    | `(240,)`        | Indices of points for the boundary conditions ($x=0$ and $x=1$)                                               |
| `ic_idx.npy`    | `int`    | `(100,)`        | Indices of points for the initial condition ($t=0$)                                                           |
| `features.npy`  | `float`  | `(1500, 50)`    | Input to the branch net, with 1500 functions and 50 features per function                                     |
| `sources.npy`   | `float`  | `(1500, 12000)` | Source fields in the PDE for the 1500 functions, which is time-independent but tiled along the time dimension |
| `solutions.npy` | `float`  | `(1500, 12000)` | Solution fields of the PDE for the 1500 functions                                                             |

If you do not have `deepxde`, you
can
download [the pre-generated dataset (130 MB)](https://drive.google.com/file/d/1WBT7nXl21fdKxedo-fbWbiCo7MOLDeeI/view?usp=sharing)
instead.

# 3. Training and metrics

### Sampling a data batch

Because this benchmark has a focus on GPU memory and time measurements, we must make sure that
the neural network see the same amount of data in one batch (namely, for one time of backprop and model update).
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

| **NAME**       | **SIZE**     | **CONTAINS**                                                                           | 
|----------------|--------------|----------------------------------------------------------------------------------------|
| `branch_input` | `(50, 50)`   | Input to the branch net, with 50 functions and 50 features per function                |
| `trunk_input`  | `(4340, 2)`  | Input to the trunk net, with 4340 points (4340 = 4000 + 240 + 100)                     |
| `source_input` | `(50, 4340)` | Source fields to be used for PDE calculation, for the 50 functions and the 4340 points |
| `u_true`       | `(50, 4340)` | Solution fields to the PDE, for the 50 functions and the 4340 points                   |

**NOTE**: the solutions `u_true` can only be used for validation (i.e., not used in training).

### Network architecture

The numbers of features for the branch and trunk nets are respectively `[50, 128, 128, 128]` and `[2, 128, 128, 128]`.
A `flax` implementation in the format of "cartesian product" is provided in [src/model.py](src/model.py),
which will be used for our baseline and ZCS solutions.
You do not need to use this implementation or the format of "cartesian product";
however, please make sure that **your network has the same amount of trainable parameters,
and it sees the same amount of data in each iteration as defined above**.

### Metrics

Report the following three numbers after **training for 1000 batches**:

* **Peak GPU memory**: we do not provide built-in code to monitor GPU usage, as one can easily do this using tools
  such as [nvitop](https://github.com/XuehaiPan/nvitop). Note that, by default, `jax` pre-allocates 75% of available GPU
  memory
  on startup, and we must disable such behaviour by `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

* **Total wall time**: because `jax` can run so fast, we exclude the time used on data sampling
  (i.e., excluding time for `DiffRecData.sample_batch`).

* **L2 relative error**: you can use `src.batched_l2_relative_error` to compute the error; after 1000 batches,
  it must come down to around 30%.

---

#### XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

The benchmark problem has been completely defined as above.

The rest is about our baseline and ZCS solutions.

#### XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

---

# 4. Baseline

We provide a baseline solution in [xtrain_baseline.py](xtrain_baseline.py):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python xtrain_baseline.py -M 50 -N 4000 
```

This solution adopts the format of "cartesian product", similar to
the classes of `PDEOperatorCartesianProd` and `DeepONetCartesianProd` in `deepxde`.
In `deepxde`, the function dimension is handled by a handmade for-loop; here we use
`vmap` to vectorise this dimension. The whole `train_step` is jit compiled.

Our measurements on a V100 GPU is reported as follows:

| METHOD   | GPU (MB) | JIT-COMPILE TIME (s) | TRAIN TIME (s) | $M=50, N=4000$ |
|----------|----------|----------------------|----------------|----------------|
| Baseline | 4955     | 3.1                  | 36.1           |                |

Now we can compare these results to the original `deepxde` solutions with the other backends
(`torch`, `tf` and `paddle`), which can be found at [DeepXDE-ZCS](https://github.com/stfc-sciml/DeepXDE-ZCS).
Clearly, this `jax` baseline has surpassed those with the other backends, meaning that
it is at least a reasonable baseline with `jax`.
However, note that the time measurements in [DeepXDE-ZCS](https://github.com/stfc-sciml/DeepXDE-ZCS) do include data
sampling.

# 5. Contributed solutions

So far we have received three solutions.
Kuangdai Leng (KL) contributed two solutions using ZCS, respectively based on `jax.grad()`
and `jax.jvp()`, and Shunyuan Mao (SM) contributed a solution only using `jax.jvp()`.

`ZCS-GRAD` (by KL):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python xtrain_zcs.py -M 50 -N 4000 
```

`ZCS-JVP` (by KL):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python xtrain_zcs_jvp.py -M 50 -N 4000 
```

`PURE-JVP` (by SM):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python xtrain_jvp.py -M 50 -N 4000 
```

The measurements on an Nvidia V100 are reported below (5 runs average). These measurements show an outstanding
reduction of GPU memory and wall time by ZCS and JVP.

| METHOD        | GPU (MB) | JIT-COMPILE TIME (s) | TRAIN TIME (s) |
|---------------|----------|----------------------|----------------|
| Baseline (KL) | 4955     | 3.1                  | 36.1           | 
| ZCS-GRAD (KL) | 603      | 3.5                  | 3.6            | 
| ZCS-JVP (KL)  | 603      | 1.4                  | 3.5            | 
| PURE-JVP (SM) | 605      | 6.6                  | 3.9            |

### 5.1 Increasing number of functions and points

We increase the problem scale by using `-M 100 -N 8000`, and the measurements are reported below (5 runs average):

| METHOD        | GPU (MB) | JIT-COMPILE TIME (s) | TRAIN TIME (s) |
|---------------|----------|----------------------|----------------|
| Baseline (KL) | 10851    | 5.4                  | 142.1          | 
| ZCS-GRAD (KL) | 867      | 3.7                  | 5.1            | 
| ZCS-JVP (KL)  | 867      | 1.5                  | 5.0            | 
| PURE-JVP (SM) | 867      | 6.8                  | 5.5            |

### 5.2 Increasing the maximum differential order

Next we change the target PDE from

$$
u_t - d u_{xx} + k u ^ 2 - f=0
$$

to

$$
u_t - d u_{xxxx} + k u ^ 2 - f=0
$$

Namely, we change the order of the spatial diffusion term from 2 to 4.
The scripts are marked by the prefix `_order4_`.
The new measurements on the same device are reported below (5 runs average, $M=50, N=4000$):

| METHOD        | GPU (MB) | JIT-COMPILE TIME (s) | TRAIN TIME (s) |
|---------------|----------|----------------------|----------------|
| ZCS-GRAD (KL) | 1115     | 5.4                  | 6.4            | 
| ZCS-JVP (KL)  | 885      | 2.5                  | 6.1            | 
| PURE-JVP (SM) | 885      | 18.6                 | 7.6            |

**NOTE**: nested `jvp` is currently unsupported by the other backends (`torch`, `tf` and `paddle`).
Therefore, `ZCS-JVP` and `PURE-JVP` cannot be extended to these backends at this moment.
