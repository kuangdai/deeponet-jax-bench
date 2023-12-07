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


class DeepONet(nn.Module):
    branch_features: tuple
    trunk_features: tuple
    cartesian_prod: bool = True

    def setup(self):
        # noinspection PyAttributeOutsideInit
        self.branch, self.trunk, self.bias = (
            FNN(self.branch_features),
            FNN(self.trunk_features),
            self.param('bias', nn.initializers.zeros, ())
        )

    def __call__(self, branch_in, trunk_in, out_channels=1):
        # forward of branch and trunk
        branch_out = self.branch(branch_in)
        trunk_out = nn.tanh(self.trunk(trunk_in))  # only trunk output is activated before einsum
        # reshape for output channels
        branch_out_channels = branch_out.reshape([branch_out.shape[0], out_channels, -1])
        if trunk_out.ndim == 1:
            # jvp case, only one point is sent
            trunk_out = jnp.expand_dims(trunk_out, axis=0)
        trunk_out_channels = trunk_out.reshape([trunk_out.shape[0], out_channels, -1])
        # this IF should NOT affect efficiency because self.cartesian_prod is constant during training
        if self.cartesian_prod:
            out = jnp.einsum("bci,nci->bnc", branch_out_channels, trunk_out_channels)
        else:
            out = jnp.einsum("Nci,Nci->Nc", branch_out_channels, trunk_out_channels)
        out += self.bias
        # if out_channels is 1, squeeze this dimension
        return out.squeeze()
