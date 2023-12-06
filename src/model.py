"""
JAX implementation of DeepONet
"""

import jax.numpy as jnp
from flax import linen as nn


class FNN(nn.Module):
    features: tuple

    def setup(self):
        # noinspection PyAttributeOutsideInit
        self.layers = [nn.Dense(name=f'dense_{i}', features=feat,
                                kernel_init=nn.initializers.glorot_normal(),
                                bias_init=nn.initializers.zeros) for
                       i, feat in enumerate(self.features[1:])]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.tanh(layer(x))
        x = self.layers[-1](x)
        return x


class DeepONetCartesianProd(nn.Module):
    branch_features: tuple
    trunk_features: tuple

    def setup(self):
        # noinspection PyAttributeOutsideInit
        self.branch, self.trunk, self.bias = (
            FNN(self.branch_features),
            FNN(self.trunk_features),
            self.param('bias', nn.initializers.zeros, ())
        )

    def __call__(self, branch_in, trunk_in):
        branch_out = self.branch(branch_in)
        # only trunk output is activated before einsum
        trunk_out = nn.tanh(self.trunk(trunk_in))
        out = jnp.einsum("bi,ni->bn", branch_out, trunk_out)
        out += self.bias
        return out
