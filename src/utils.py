"""
utils
"""

import jax.numpy as jnp
import numpy as np


def mse_to_zeros(x):
    """ mse with truth being zeros """
    return jnp.mean((x - jnp.zeros_like(x)) ** 2)


def batched_l2_relative_error(y_true, y_pred):
    """ batched l2 relative error """
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1))


class DiffRecData:
    """ convenient data access """

    def __init__(self, data_path):
        self.xt_all = np.load(f'{data_path}/xt_all.npy')
        self.bc_idx = np.load(f'{data_path}/bc_idx.npy')
        self.ic_idx = np.load(f'{data_path}/ic_idx.npy')
        self.features = np.load(f'{data_path}/features.npy')
        self.sources = np.load(f'{data_path}/sources.npy')
        self.solutions = np.load(f'{data_path}/solutions.npy')
        self.n_features = self.features.shape[1]
        self.n_dims = self.xt_all.shape[1]
        self.n_bc = len(self.bc_idx)
        self.n_ic = len(self.ic_idx)
        self.train_split = 1000

    def sample_batch(self, n_funcs, n_points, seed=0, train=True):
        np.random.seed(seed)
        if train:
            func_range = range(0, self.train_split)
        else:
            func_range = range(self.train_split, len(self.features))
        # sample features
        func_idx = np.random.choice(func_range, n_funcs, replace=False)
        branch_input = self.features[func_idx]
        # sample points
        pde_idx = np.random.choice(len(self.xt_all), n_points, replace=False)
        all_idx = np.concatenate((pde_idx, self.bc_idx, self.ic_idx))
        trunk_input = self.xt_all[all_idx]
        # sample sources and solutions
        source_input = self.sources[func_idx][:, all_idx]
        solutions = self.solutions[func_idx][:, all_idx]
        return (jnp.array(branch_input), jnp.array(trunk_input),
                jnp.array(source_input), jnp.array(solutions))
