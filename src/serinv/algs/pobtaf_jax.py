# Copyright 2023-2025 ETH Zurich. All rights reserved.

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax
from functools import partial


# JAX cholesky with lower=True
_jax_cholesky = partial(jax.scipy.linalg.cholesky, lower=True)


def _cholesky_grad(L, bar_L):
    """
    Compute gradient of Cholesky factorization using JAX's VJP.

    Given L = chol(A) and bar_L (cotangent w.r.t. L), computes bar_A (cotangent w.r.t. A).
    Uses JAX's internal VJP for correctness.
    """
    A = L @ L.conj().T
    _, vjp_fn = jax.vjp(lambda x: jax.scipy.linalg.cholesky(x, lower=True), A)
    (bar_A,) = vjp_fn(bar_L)
    return bar_A


def _pobtaf_impl(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
    A_lower_arrow_blocks,
    A_arrow_tip_block,
    factorize_last_block=True,
):
    """
    Internal implementation of BTA Cholesky factorization.
    This is called by both the forward pass and for comparison with autodiff.
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


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def pobtaf_jax_optimized(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
    A_lower_arrow_blocks,
    A_arrow_tip_block,
    factorize_last_block=True,
):
    """
    JAX-optimized Cholesky factorization for block-tridiagonal-arrowhead matrices.

    Uses custom VJP for efficient backward pass.
    """
    return _pobtaf_impl(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        factorize_last_block,
    )


def _pobtaf_fwd(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
    A_lower_arrow_blocks,
    A_arrow_tip_block,
    factorize_last_block,
):
    """Forward pass: compute outputs and save residuals for backward pass."""
    L_diag, L_lower_diag, L_lower_arrow, L_tip = _pobtaf_impl(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        factorize_last_block,
    )
    outputs = (L_diag, L_lower_diag, L_lower_arrow, L_tip)
    residuals = (L_diag, L_lower_diag, L_lower_arrow, L_tip)
    return outputs, residuals


def _pobtaf_bwd(factorize_last_block, residuals, g):
    """
    Backward pass for BTA Cholesky factorization.

    Propagates cotangents from outputs (L factors) to inputs (A blocks).
    """
    L_diag, L_lower_diag, L_lower_arrow, L_tip = residuals
    bar_L_diag, bar_L_lower_diag, bar_L_lower_arrow, bar_L_tip = g

    n_diag_blocks = L_diag.shape[0]

    # Initialize output cotangents
    bar_A_diag = jnp.zeros_like(L_diag)
    bar_A_lower_diag = jnp.zeros_like(L_lower_diag)
    bar_A_lower_arrow = jnp.zeros_like(L_lower_arrow)
    bar_A_tip = jnp.zeros_like(L_tip)

    # Read-only cotangent snapshots — these are only indexed (never
    # written) inside the backward loop, so we keep them out of the
    # fori_loop carry to save ~30 GB of duplicated state.
    bar_L_diag_snap = bar_L_diag
    bar_L_lower_arrow_snap = bar_L_lower_arrow

    # Accumulated cotangent for intermediate A_tip (tracks Schur complement contributions)
    bar_A_tip_accum = jnp.zeros_like(L_tip)

    if factorize_last_block:
        # Backward through: L_tip = chol(A_tip_final)
        bar_A_tip_accum = _cholesky_grad(L_tip, bar_L_tip)

        L_nn = L_diag[-1]
        L_arrow_n = L_lower_arrow[-1]

        # Backward through: A_tip_final = A_tip_prev - L_arrow_n @ L_arrow_n^T
        bar_L_arrow_n = bar_L_lower_arrow_snap[-1] - (bar_A_tip_accum + bar_A_tip_accum.conj().T) @ L_arrow_n

        # Backward through: L_arrow_n = A_arrow_n @ L_nn^{-T}
        bar_A_lower_arrow = bar_A_lower_arrow.at[-1].set(
            jax.scipy.linalg.solve_triangular(
                L_nn.conj().T, bar_L_arrow_n.conj().T, lower=False
            ).conj().T
        )

        # Contribution to bar_L_nn from triangular solve
        temp_arrow_n = jax.scipy.linalg.solve_triangular(
            L_nn.conj().T, bar_L_arrow_n.conj().T, lower=False
        )
        bar_L_nn_from_solve = -temp_arrow_n @ L_arrow_n
        # Update the snapshot at the last block for the loop below
        bar_L_diag_snap = bar_L_diag_snap.at[-1].add(bar_L_nn_from_solve)

        # Backward through: L_nn = chol(A_nn_modified)
        bar_A_diag = bar_A_diag.at[-1].set(_cholesky_grad(L_nn, bar_L_diag_snap[-1]))

    # Handle n_diag_blocks == 1 case (no loop iterations)
    if n_diag_blocks == 1:
        bar_A_tip = bar_A_tip_accum
        return (bar_A_diag, bar_A_lower_diag, bar_A_lower_arrow, bar_A_tip)

    # Backward loop: i = n_diag_blocks-2 down to 0
    # bar_L_diag_snap and bar_L_lower_arrow_snap are captured via
    # closure — they are read-only inside the loop body.
    def bwd_body_fn(i_rev, state):
        (bar_A_diag, bar_A_lower_diag, bar_A_lower_arrow,
         bar_A_tip_accum) = state

        i = n_diag_blocks - 2 - i_rev

        L_ii = L_diag[i]
        L_ip1_i = L_lower_diag[i]
        L_arrow_i = L_lower_arrow[i]

        bar_A_diag_ip1 = bar_A_diag[i + 1]

        # Backward through: A_diag[i+1] -= L_ip1_i @ L_ip1_i^T
        bar_L_ip1_i = bar_L_lower_diag[i] - (bar_A_diag_ip1 + bar_A_diag_ip1.conj().T) @ L_ip1_i

        # Backward through: A_lower_arrow[i+1] -= L_arrow_i @ L_ip1_i^H
        bar_A_lower_arrow_ip1 = bar_A_lower_arrow[i + 1]
        bar_L_arrow_i = bar_L_lower_arrow_snap[i] - bar_A_lower_arrow_ip1 @ L_ip1_i
        bar_L_ip1_i = bar_L_ip1_i - bar_A_lower_arrow_ip1.conj().T @ L_arrow_i

        # Backward through: A_tip -= L_arrow_i @ L_arrow_i^T
        bar_L_arrow_i = bar_L_arrow_i - (bar_A_tip_accum + bar_A_tip_accum.conj().T) @ L_arrow_i

        # Backward through: L_ip1_i = A_ip1_i @ L_ii^{-T}
        bar_A_lower_diag = bar_A_lower_diag.at[i].set(
            jax.scipy.linalg.solve_triangular(
                L_ii.conj().T, bar_L_ip1_i.conj().T, lower=False
            ).conj().T
        )

        # Backward through: L_arrow_i = A_arrow_i @ L_ii^{-T}
        bar_A_lower_arrow = bar_A_lower_arrow.at[i].set(
            jax.scipy.linalg.solve_triangular(
                L_ii.conj().T, bar_L_arrow_i.conj().T, lower=False
            ).conj().T
        )

        # Contributions to bar_L_ii from triangular solves
        temp_lower = jax.scipy.linalg.solve_triangular(
            L_ii.conj().T, bar_L_ip1_i.conj().T, lower=False
        )
        bar_L_ii_from_lower = -temp_lower @ L_ip1_i

        temp_arrow = jax.scipy.linalg.solve_triangular(
            L_ii.conj().T, bar_L_arrow_i.conj().T, lower=False
        )
        bar_L_ii_from_arrow = -temp_arrow @ L_arrow_i

        bar_L_ii_total = bar_L_diag_snap[i] + bar_L_ii_from_lower + bar_L_ii_from_arrow

        # Backward through: L_ii = chol(A_ii)
        bar_A_diag = bar_A_diag.at[i].set(_cholesky_grad(L_ii, bar_L_ii_total))

        return (bar_A_diag, bar_A_lower_diag, bar_A_lower_arrow,
                bar_A_tip_accum)

    init_state = (bar_A_diag, bar_A_lower_diag, bar_A_lower_arrow,
                  bar_A_tip_accum)

    final_state = lax.fori_loop(0, n_diag_blocks - 1, bwd_body_fn, init_state)
    bar_A_diag, bar_A_lower_diag, bar_A_lower_arrow, bar_A_tip_accum = final_state

    # Final bar_A_tip is the accumulated cotangent
    bar_A_tip = bar_A_tip_accum

    return (bar_A_diag, bar_A_lower_diag, bar_A_lower_arrow, bar_A_tip)


pobtaf_jax_optimized.defvjp(_pobtaf_fwd, _pobtaf_bwd)
