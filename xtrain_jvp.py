"""
Training with jax.jvp, by Shunyuan Mao
"""

import argparse
import functools
from functools import partial
from time import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange

from src.model import DeepONet
from src.utils import mse_to_zeros, batched_l2_relative_error, DiffRecData


def compute_u_pde(forward_fn, branch_input, trunk_input, source_input):
    """ diffusion-reaction equation with VMAP over function dimension """
    branch_tangent = jnp.zeros(branch_input.shape)
    trunk_tangent_x = jnp.array([1., 0.])
    trunk_tangent_t = jnp.array([0., 1.])

    @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def get_u_and_u_t(branch_input_, trunk_input_):
        """ get u and u_t """
        return jax.jvp(forward_fn, (branch_input_, trunk_input_), (branch_tangent, trunk_tangent_t))

    def get_u_x(branch_input_, trunk_input_):
        return jax.jvp(forward_fn, (branch_input_, trunk_input_), (branch_tangent, trunk_tangent_x))[1]

    @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def get_u_xx(branch_input_, trunk_input_):
        return jax.jvp(get_u_x, (branch_input_, trunk_input_), (branch_tangent, trunk_tangent_x))[1]

    # constants
    d, k = 0.01, 0.01

    # forward
    u, u_t = get_u_and_u_t(branch_input, trunk_input)
    u_xx = get_u_xx(branch_input, trunk_input)
    pde = u_t - d * u_xx + k * u ** 2 - source_input[:, :, None]
    return u, pde


@partial(jax.jit, static_argnames=['n_points_pde', 'n_points_bc', 'model_interface'])
def train_step(state, branch_input, trunk_input, source_input,
               n_points_pde, n_points_bc, model_interface):
    """ train for a single step """

    def loss_fn(params_):
        """ loss function, for AD w.r.t. network weights """

        def forward_fn(branch_input_, trunk_input_):
            """ forward function, for AD w.r.t. coordinates """
            return model_interface.apply({'params': params_},
                                         branch_in=branch_input_, trunk_in=trunk_input_)

        u_val, pde_val = compute_u_pde(forward_fn, branch_input, trunk_input, source_input)
        pde_loss_ = mse_to_zeros(pde_val[:, :n_points_pde])
        bc_loss_ = mse_to_zeros(u_val[:, n_points_pde:n_points_pde + n_points_bc])
        ic_loss_ = mse_to_zeros(u_val[:, n_points_pde + n_points_bc:])
        return pde_loss_ + bc_loss_ + ic_loss_, (pde_loss_, bc_loss_, ic_loss_)

    # loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (pde_loss, bc_loss, ic_loss)), grads = grad_fn(state.params)
    # update
    state = state.apply_gradients(grads=grads)
    return state, loss, pde_loss, bc_loss, ic_loss


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--iterations', type=int, default=1000,
                        help='number of iterations')
    parser.add_argument('-M', '--n-functions', type=int, default=50,
                        help='number of functions in a batch')
    parser.add_argument('-N', '--n-points', type=int, default=4000,
                        help='number of collocation points in a batch')
    args = parser.parse_args()

    # load data
    data = DiffRecData(data_path='./data')

    # number of functions and points in a batch
    # Note: the default values come from the diff_rec example in DeepXDE-ZCS
    #       for comparison with pytorch, tensorflow and paddle
    N_FUNCTIONS = args.n_functions  # noqa
    N_POINTS_PDE = args.n_points  # noqa

    # train state
    model = DeepONet(branch_features=(data.n_features, 128, 128, 128),
                     trunk_features=(data.n_dims, 128, 128, 128),
                     cartesian_prod=True)
    branch_in, trunk_in, _, _ = data.sample_batch(N_FUNCTIONS, N_POINTS_PDE)
    params = model.init(jax.random.PRNGKey(0), branch_in, trunk_in)['params']
    optimizer = optax.adam(learning_rate=0.0005)
    the_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

    #################
    # training loop #
    #################
    pbar = trange(args.iterations, desc='Training')
    t_total = 0.
    for it in pbar:
        # sample data
        branch_in, trunk_in, source_in, _ = data.sample_batch(
            N_FUNCTIONS, N_POINTS_PDE, seed=it, train=True)

        # update
        t0 = time()  # wall time excluding data sampling
        the_state, loss_val, pde_loss_val, bc_loss_val, ic_loss_val = \
            train_step(the_state, branch_in, trunk_in, source_in,
                       N_POINTS_PDE, data.n_bc, model)

        pbar.set_postfix_str(f"L_pde={pde_loss_val:.4e}, "
                             f"L_bc={bc_loss_val:.4e}, "
                             f"L_ic={ic_loss_val:.4e}, "
                             f"L={loss_val:.4e}")
        t_total += time() - t0
    print(f'Training done in {t_total:.1f} seconds')

    ##############
    # evaluation #
    ##############
    # sample solution
    print('Validating against true solution...')
    n_functions_eval, n_points_eval = 200, 10000
    branch_in, trunk_in, source_in, u_true = data.sample_batch(
        n_functions_eval, n_points_eval, train=False)
    # prediction
    u_pred = model.apply({'params': the_state.params}, branch_in=branch_in, trunk_in=trunk_in)
    print(f'L2 relative error: {batched_l2_relative_error(u_true, u_pred)}')


if __name__ == "__main__":
    run()
