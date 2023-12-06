"""
prepare data
"""

from pathlib import Path

import deepxde as dde  # only to use GRF
import numpy as np
from tqdm import tqdm

from src.solution import solve_ADR

if __name__ == '__main__':
    """ generate data """
    # seed
    np.random.seed(0)

    # point sampling over a 100x120 meshgrid
    nx, nt = 100, 120
    x = np.linspace(0., 1., nx)
    t = np.linspace(0., 1., nt)
    xt = np.stack(np.meshgrid(x, t, indexing='ij'), axis=-1)
    xt_all = xt.reshape((-1, 2))
    bc_idx = np.concatenate((np.where(np.isclose(xt_all[:, 0], 0.))[0],
                             np.where(np.isclose(xt_all[:, 0], 1.))[0]))
    ic_idx = np.where(np.isclose(xt_all[:, 1], 0.))[0]
    assert len(bc_idx) == nt * 2 and len(ic_idx) == nx

    # features and sources
    num_func = 1500  # 1000 for training, 500 for testing and validation
    func_space = dde.data.GRF(length_scale=0.2)
    implicits = func_space.random(num_func)
    features = func_space.eval_batch(implicits, np.linspace(0., 1., 50))
    sources = func_space.eval_batch(implicits, xt_all[:, 0])

    # true solutions
    u_true_list = []
    for source in tqdm(sources, desc='Preparing solutions'):
        f = source.reshape(nx, nt)[:, 0]
        _, _, u_true = solve_ADR(
            xmin=0., xmax=1., tmin=0., tmax=1.,
            k=lambda x_: 0.01 * np.ones_like(x_),
            v=lambda x_: np.zeros_like(x_),
            g=lambda u_: 0.01 * u_ ** 2,
            dg=lambda u_: 0.02 * u_,
            f=lambda x_, t_: np.tile(f[:, None], reps=(1, len(t_))),
            u0=lambda x_: np.zeros_like(x_),
            Nx=nx, Nt=nt)
        u_true_list.append(u_true.reshape(-1))
    solutions = np.stack(u_true_list, axis=0)

    # info
    print('Data generated:')
    print('* Points: ')
    print(f'  in domain:   {xt_all.shape}')
    print(f'  on boundary: {bc_idx.shape}')
    print(f'  on initial:  {ic_idx.shape}')
    print('* Functions: ')
    print(f'  features:    {features.shape}')
    print(f'  sources:     {sources.shape}')
    print(f'  solutions:   {solutions.shape}')

    # save
    path = Path('data')
    path.mkdir(exist_ok=True)
    np.save(path / 'xt_all', xt_all)
    np.save(path / 'bc_idx', bc_idx)
    np.save(path / 'ic_idx', ic_idx)
    np.save(path / 'features', features)
    np.save(path / 'sources', sources)
    np.save(path / 'solutions', solutions)
