# Copyright 2023-2025 ETH Zurich. All rights reserved.

"""
Optimized JAX implementation of block-tridiagonal Cholesky factorization.
This is a simplified version of pobtaf_jax without the arrowhead.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax
from functools import partial


_jax_cholesky = partial(jax.scipy.linalg.cholesky, lower=True)


def _pobtf_impl(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
):
    """
    Internal implementation of BT (block-tridiagonal) Cholesky factorization.

    Parameters
    ----------
    A_diagonal_blocks : array (n_blocks, block_size, block_size)
        Diagonal blocks of the matrix
    A_lower_diagonal_blocks : array (n_blocks-1, block_size, block_size)
        Lower diagonal blocks

    Returns
    -------
    L_diagonal_blocks : Cholesky factor diagonal blocks
    L_lower_diagonal_blocks : Cholesky factor lower diagonal blocks
    """
    cholesky = _jax_cholesky
    n_diag_blocks = A_diagonal_blocks.shape[0]

    if n_diag_blocks == 1:
        L_diag = cholesky(A_diagonal_blocks[0])
        return A_diagonal_blocks.at[0].set(L_diag), A_lower_diagonal_blocks

    init_state = (A_diagonal_blocks, A_lower_diagonal_blocks)

    def body_fn(i, state):
        """Loop body for forward block-Cholesky."""
        diag_blocks, lower_diag_blocks = state

        L_ii = cholesky(diag_blocks[i])
        diag_blocks = diag_blocks.at[i].set(L_ii)

        L_ip1_i = jax.scipy.linalg.solve_triangular(
            L_ii,
            lower_diag_blocks[i].conj().T,
            lower=True,
        ).conj().T
        lower_diag_blocks = lower_diag_blocks.at[i].set(L_ip1_i)

        diag_blocks = diag_blocks.at[i + 1].add(-L_ip1_i @ L_ip1_i.conj().T)

        return (diag_blocks, lower_diag_blocks)

    diag_blocks, lower_diag_blocks = lax.fori_loop(
        0, n_diag_blocks - 1, body_fn, init_state
    )

    L_nn = cholesky(diag_blocks[-1])
    diag_blocks = diag_blocks.at[-1].set(L_nn)

    return diag_blocks, lower_diag_blocks


@jax.jit
def pobtf_jax(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
):
    """
    JAX block-tridiagonal Cholesky factorization.

    Computes L such that A = L @ L^T for a block-tridiagonal matrix A.

    Parameters
    ----------
    A_diagonal_blocks : array (n_blocks, block_size, block_size)
        Diagonal blocks of the SPD matrix
    A_lower_diagonal_blocks : array (n_blocks-1, block_size, block_size)
        Lower diagonal blocks

    Returns
    -------
    L_diagonal_blocks : Cholesky factor diagonal blocks
    L_lower_diagonal_blocks : Cholesky factor lower diagonal blocks
    """
    return _pobtf_impl(A_diagonal_blocks, A_lower_diagonal_blocks)


def compute_logdet_bt_jax(L_diagonal_blocks):
    """
    Compute log determinant from BT Cholesky factor.

    Parameters
    ----------
    L_diagonal_blocks : array (n_blocks, block_size, block_size)
        Diagonal blocks of Cholesky factor

    Returns
    -------
    logdet : scalar
        log|A| = 2 * sum(log(diag(L)))
    """
    diag_vals = jnp.diagonal(L_diagonal_blocks, axis1=1, axis2=2)
    return 2.0 * jnp.sum(jnp.log(diag_vals))


@jax.jit
def pobtf_logdet_jax(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
):
    """
    Compute Cholesky factorization and log determinant in one pass.

    This is optimized for when you only need the logdet, not the factors.

    Parameters
    ----------
    A_diagonal_blocks : array (n_blocks, block_size, block_size)
    A_lower_diagonal_blocks : array (n_blocks-1, block_size, block_size)

    Returns
    -------
    logdet : scalar
        log|A|
    """
    L_diag, _ = _pobtf_impl(A_diagonal_blocks, A_lower_diagonal_blocks)
    return compute_logdet_bt_jax(L_diag)
