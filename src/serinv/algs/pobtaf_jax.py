# Copyright 2023-2025 ETH Zurich. All rights reserved.

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax
from functools import partial


# JAX cholesky with lower=True
_jax_cholesky = partial(jax.scipy.linalg.cholesky, lower=True)


def pobtaf_jax_optimized(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
    A_lower_arrow_blocks,
    A_arrow_tip_block,
    factorize_last_block=True,
):
    """
    JAX-optimized Cholesky factorization using lax.fori_loop.
    """
    cholesky = _jax_cholesky
    n_diag_blocks = A_diagonal_blocks.shape[0]

    # Handle n_diag_blocks=1 special case (no loop iterations needed)
    if n_diag_blocks == 1:
        diag_blocks = A_diagonal_blocks
        lower_diag_blocks = A_lower_diagonal_blocks
        lower_arrow_blocks = A_lower_arrow_blocks
        arrow_tip = A_arrow_tip_block

        if factorize_last_block:
            L_nn = cholesky(diag_blocks[0])
            diag_blocks = diag_blocks.at[0].set(L_nn)

            L_arrow_n = jax.scipy.linalg.solve_triangular(
                L_nn,
                lower_arrow_blocks[0].conj().T,
                lower=True,
            ).conj().T
            lower_arrow_blocks = lower_arrow_blocks.at[0].set(L_arrow_n)

            arrow_tip = arrow_tip - L_arrow_n @ L_arrow_n.conj().T
            L_tip = cholesky(arrow_tip)
            arrow_tip = L_tip

        return diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip

    # State tuple: (diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip)
    init_state = (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
    )

    def body_fn(i, state):
        """Loop body for forward block-Cholesky."""
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip = state

        # L_{i, i} = chol(A_{i, i})
        L_ii = cholesky(diag_blocks[i])
        diag_blocks = diag_blocks.at[i].set(L_ii)

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_ip1_i = jax.scipy.linalg.solve_triangular(
            L_ii,
            lower_diag_blocks[i].conj().T,
            lower=True,
        ).conj().T
        lower_diag_blocks = lower_diag_blocks.at[i].set(L_ip1_i)

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_ii,
            lower_arrow_blocks[i].conj().T,
            lower=True,
        ).conj().T
        lower_arrow_blocks = lower_arrow_blocks.at[i].set(L_arrow_i)

        # Update next diagonal block
        diag_blocks = diag_blocks.at[i + 1].add(
            -L_ip1_i @ L_ip1_i.conj().T
        )

        # Update next arrow block
        lower_arrow_blocks = lower_arrow_blocks.at[i + 1].add(
            -L_arrow_i @ L_ip1_i.conj().T
        )

        # Update arrow tip
        arrow_tip = arrow_tip - L_arrow_i @ L_arrow_i.conj().T

        return (diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip)

    # Run the loop
    diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip = lax.fori_loop(
        0, n_diag_blocks - 1, body_fn, init_state
    )

    if factorize_last_block:
        # L_{ndb, ndb} = chol(A_{ndb, ndb})
        L_nn = cholesky(diag_blocks[-1])
        diag_blocks = diag_blocks.at[-1].set(L_nn)

        # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
        L_arrow_n = jax.scipy.linalg.solve_triangular(
            L_nn,
            lower_arrow_blocks[-1].conj().T,
            lower=True,
        ).conj().T
        lower_arrow_blocks = lower_arrow_blocks.at[-1].set(L_arrow_n)

        # Update arrow tip
        arrow_tip = arrow_tip - L_arrow_n @ L_arrow_n.conj().T

        # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
        L_tip = cholesky(arrow_tip)
        arrow_tip = L_tip

    # Modify arrays in place
    A_diagonal_blocks = diag_blocks
    A_lower_diagonal_blocks = lower_diag_blocks
    A_lower_arrow_blocks = lower_arrow_blocks
    A_arrow_tip_block = arrow_tip

    return A_diagonal_blocks, A_lower_diagonal_blocks, A_lower_arrow_blocks, A_arrow_tip_block
