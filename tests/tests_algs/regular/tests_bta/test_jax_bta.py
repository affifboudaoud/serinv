# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tests.testing_utils import bta_dense_to_arrays, dd_bta, symmetrize
from serinv.algs.pobtaf import _pobtaf
from serinv.algs.pobtaf_jax import pobtaf_jax_optimized
from serinv.algs.pobtas import _pobtas
from serinv.algs.pobtas_jax import pobtas_jax_optimized


@pytest.mark.mpi_skip()
def test_pobtaf_jax_matches_numpy(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    dtype: np.dtype,
):
    """Test that JAX-optimized pobtaf produces same results as NumPy version."""

    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=False,
        dtype=dtype,
    )
    symmetrize(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_lower_arrow_blocks,
        _,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    np_diag = A_diagonal_blocks.copy()
    np_lower_diag = A_lower_diagonal_blocks.copy()
    np_lower_arrow = A_lower_arrow_blocks.copy()
    np_arrow_tip = A_arrow_tip_block.copy()

    _pobtaf(np_diag, np_lower_diag, np_lower_arrow, np_arrow_tip, factorize_last_block=True)

    jax_diag = jnp.array(A_diagonal_blocks)
    jax_lower_diag = jnp.array(A_lower_diagonal_blocks)
    jax_lower_arrow = jnp.array(A_lower_arrow_blocks)
    jax_arrow_tip = jnp.array(A_arrow_tip_block)

    jax_result = pobtaf_jax_optimized(
        jax_diag, jax_lower_diag, jax_lower_arrow, jax_arrow_tip, factorize_last_block=True
    )
    jax_diag_out, jax_lower_diag_out, jax_lower_arrow_out, jax_arrow_tip_out = jax_result

    assert np.allclose(np_diag, np.array(jax_diag_out), rtol=1e-10, atol=1e-12), \
        f"Diagonal blocks mismatch: max diff = {np.max(np.abs(np_diag - np.array(jax_diag_out)))}"
    assert np.allclose(np_lower_diag, np.array(jax_lower_diag_out), rtol=1e-10, atol=1e-12), \
        f"Lower diagonal blocks mismatch: max diff = {np.max(np.abs(np_lower_diag - np.array(jax_lower_diag_out)))}"
    assert np.allclose(np_lower_arrow, np.array(jax_lower_arrow_out), rtol=1e-10, atol=1e-12), \
        f"Lower arrow blocks mismatch: max diff = {np.max(np.abs(np_lower_arrow - np.array(jax_lower_arrow_out)))}"
    assert np.allclose(np_arrow_tip, np.array(jax_arrow_tip_out), rtol=1e-10, atol=1e-12), \
        f"Arrow tip block mismatch: max diff = {np.max(np.abs(np_arrow_tip - np.array(jax_arrow_tip_out)))}"


@pytest.mark.mpi_skip()
def test_pobtas_jax_matches_numpy(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    dtype: np.dtype,
):
    """Test that JAX-optimized pobtas produces same results as NumPy version."""

    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=False,
        dtype=dtype,
    )
    symmetrize(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_lower_arrow_blocks,
        _,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    L_diag = A_diagonal_blocks.copy()
    L_lower_diag = A_lower_diagonal_blocks.copy()
    L_lower_arrow = A_lower_arrow_blocks.copy()
    L_arrow_tip = A_arrow_tip_block.copy()
    _pobtaf(L_diag, L_lower_diag, L_lower_arrow, L_arrow_tip, factorize_last_block=True)

    np.random.seed(42)
    total_size = n_diag_blocks * diagonal_blocksize + arrowhead_blocksize
    if dtype == "complex128":
        rhs = np.random.rand(total_size) + 1j * np.random.rand(total_size)
    else:
        rhs = np.random.rand(total_size)

    np_rhs_fwd = rhs.copy()
    _pobtas(L_diag, L_lower_diag, L_lower_arrow, L_arrow_tip, np_rhs_fwd, trans="N", partial=False)

    np_rhs_bwd = np_rhs_fwd.copy()
    _pobtas(L_diag, L_lower_diag, L_lower_arrow, L_arrow_tip, np_rhs_bwd, trans="C", partial=False)

    jax_L_diag, jax_L_lower_diag, jax_L_lower_arrow, jax_L_arrow_tip = pobtaf_jax_optimized(
        jnp.array(A_diagonal_blocks),
        jnp.array(A_lower_diagonal_blocks),
        jnp.array(A_lower_arrow_blocks),
        jnp.array(A_arrow_tip_block),
        factorize_last_block=True,
    )

    jax_rhs_fwd = pobtas_jax_optimized(
        jax_L_diag, jax_L_lower_diag, jax_L_lower_arrow, jax_L_arrow_tip,
        jnp.array(rhs), trans="N", partial=False
    )

    jax_rhs_bwd = pobtas_jax_optimized(
        jax_L_diag, jax_L_lower_diag, jax_L_lower_arrow, jax_L_arrow_tip,
        jax_rhs_fwd, trans="C", partial=False
    )

    assert np.allclose(np_rhs_fwd, np.array(jax_rhs_fwd), rtol=1e-10, atol=1e-12), \
        f"Forward solve mismatch: max diff = {np.max(np.abs(np_rhs_fwd - np.array(jax_rhs_fwd)))}"

    assert np.allclose(np_rhs_bwd, np.array(jax_rhs_bwd), rtol=1e-10, atol=1e-12), \
        f"Backward solve mismatch: max diff = {np.max(np.abs(np_rhs_bwd - np.array(jax_rhs_bwd)))}"


@pytest.mark.mpi_skip()
def test_jax_full_solve_matches_numpy(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    dtype: np.dtype,
):
    """Test that full solve (L L^T x = b) produces same results for JAX and NumPy."""

    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=False,
        dtype=dtype,
    )
    symmetrize(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_lower_arrow_blocks,
        _,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    np.random.seed(42)
    total_size = n_diag_blocks * diagonal_blocksize + arrowhead_blocksize
    if dtype == "complex128":
        rhs = np.random.rand(total_size) + 1j * np.random.rand(total_size)
    else:
        rhs = np.random.rand(total_size)

    np_L_diag = A_diagonal_blocks.copy()
    np_L_lower_diag = A_lower_diagonal_blocks.copy()
    np_L_lower_arrow = A_lower_arrow_blocks.copy()
    np_L_arrow_tip = A_arrow_tip_block.copy()
    _pobtaf(np_L_diag, np_L_lower_diag, np_L_lower_arrow, np_L_arrow_tip, factorize_last_block=True)

    np_solution = rhs.copy()
    _pobtas(np_L_diag, np_L_lower_diag, np_L_lower_arrow, np_L_arrow_tip, np_solution, trans="N", partial=False)
    _pobtas(np_L_diag, np_L_lower_diag, np_L_lower_arrow, np_L_arrow_tip, np_solution, trans="C", partial=False)

    jax_L_diag, jax_L_lower_diag, jax_L_lower_arrow, jax_L_arrow_tip = pobtaf_jax_optimized(
        jnp.array(A_diagonal_blocks),
        jnp.array(A_lower_diagonal_blocks),
        jnp.array(A_lower_arrow_blocks),
        jnp.array(A_arrow_tip_block),
        factorize_last_block=True,
    )

    jax_solution = pobtas_jax_optimized(
        jax_L_diag, jax_L_lower_diag, jax_L_lower_arrow, jax_L_arrow_tip,
        jnp.array(rhs), trans="N", partial=False
    )
    jax_solution = pobtas_jax_optimized(
        jax_L_diag, jax_L_lower_diag, jax_L_lower_arrow, jax_L_arrow_tip,
        jax_solution, trans="C", partial=False
    )

    assert np.allclose(np_solution, np.array(jax_solution), rtol=1e-10, atol=1e-12), \
        f"Full solve mismatch: max diff = {np.max(np.abs(np_solution - np.array(jax_solution)))}"

    x_ref = np.linalg.solve(A, rhs)
    assert np.allclose(np_solution, x_ref, rtol=1e-8, atol=1e-10), \
        f"NumPy solution doesn't match reference: max diff = {np.max(np.abs(np_solution - x_ref))}"
    assert np.allclose(np.array(jax_solution), x_ref, rtol=1e-8, atol=1e-10), \
        f"JAX solution doesn't match reference: max diff = {np.max(np.abs(np.array(jax_solution) - x_ref))}"


@pytest.mark.mpi_skip()
def test_jax_autodiff_through_factorization(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    dtype: np.dtype,
):
    """Test that JAX autodiff works through the factorization."""

    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=False,
        dtype=dtype,
    )
    symmetrize(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_lower_arrow_blocks,
        _,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    jax_diag = jnp.array(A_diagonal_blocks)
    jax_lower_diag = jnp.array(A_lower_diagonal_blocks)
    jax_lower_arrow = jnp.array(A_lower_arrow_blocks)
    jax_arrow_tip = jnp.array(A_arrow_tip_block)

    def objective(scale):
        diag_scaled = jax_diag * scale
        L_diag, L_lower_diag, L_lower_arrow, L_arrow_tip = pobtaf_jax_optimized(
            diag_scaled, jax_lower_diag.copy(), jax_lower_arrow.copy(), jax_arrow_tip.copy(),
            factorize_last_block=True,
        )
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.array([L_diag[i, j, j] for i in range(L_diag.shape[0]) for j in range(L_diag.shape[1])]))))
        logdet += 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L_arrow_tip))))
        return logdet.real

    grad_func = jax.grad(objective)
    scale = 1.0
    grad = grad_func(scale)

    assert jnp.isfinite(grad), "Gradient is not finite"
    assert grad != 0.0, "Gradient is zero (unexpected)"

    eps = 1e-6
    fd_grad = (objective(scale + eps) - objective(scale - eps)) / (2 * eps)

    rel_error = jnp.abs(grad - fd_grad) / (jnp.abs(fd_grad) + 1e-10)
    assert rel_error < 1e-4, f"Gradient mismatch: autodiff={grad}, fd={fd_grad}, rel_error={rel_error}"
