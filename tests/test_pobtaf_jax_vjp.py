# Copyright 2023-2025 ETH Zurich. All rights reserved.

"""
Tests for pobtaf_jax_optimized gradient correctness.

This test verifies that JAX autodiff through the BTA Cholesky factorization
produces correct gradients, validated against finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def generate_spd_block(n, seed=None):
    """Generate a symmetric positive definite block."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    return jnp.array(A @ A.T + n * np.eye(n))


def generate_bta_matrix(n_blocks, block_size, arrow_size, seed=42):
    """Generate a random SPD block-tridiagonal-arrowhead matrix."""
    np.random.seed(seed)

    diag_blocks = []
    for i in range(n_blocks):
        diag_blocks.append(generate_spd_block(block_size, seed=seed + i))
    diag_blocks = jnp.stack(diag_blocks)

    lower_diag_blocks = []
    for i in range(n_blocks - 1):
        lower_diag_blocks.append(
            jnp.array(0.1 * np.random.randn(block_size, block_size))
        )
    if n_blocks > 1:
        lower_diag_blocks = jnp.stack(lower_diag_blocks)
    else:
        lower_diag_blocks = jnp.zeros((0, block_size, block_size))

    lower_arrow_blocks = []
    for i in range(n_blocks):
        lower_arrow_blocks.append(
            jnp.array(0.1 * np.random.randn(arrow_size, block_size))
        )
    lower_arrow_blocks = jnp.stack(lower_arrow_blocks)

    arrow_tip = generate_spd_block(arrow_size, seed=seed + 100)

    return diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip


def test_forward_pass_produces_valid_factors():
    """Test that the forward pass produces valid Cholesky factors."""
    from serinv.algs.pobtaf_jax import pobtaf_jax_optimized

    n_blocks, block_size, arrow_size = 3, 4, 2
    A_diag, A_lower_diag, A_lower_arrow, A_tip = generate_bta_matrix(
        n_blocks, block_size, arrow_size
    )

    L_diag, L_lower_diag, L_lower_arrow, L_tip = pobtaf_jax_optimized(
        A_diag, A_lower_diag, A_lower_arrow, A_tip
    )

    assert L_diag.shape == A_diag.shape
    assert L_lower_diag.shape == A_lower_diag.shape
    assert L_lower_arrow.shape == A_lower_arrow.shape
    assert L_tip.shape == A_tip.shape

    for i in range(n_blocks):
        assert jnp.allclose(L_diag[i], jnp.tril(L_diag[i])), f"L_diag[{i}] not lower triangular"


def test_gradient_with_finite_differences():
    """Test JAX autodiff gradients against finite differences.

    Note: A_diag and A_tip are symmetric matrices, so we use symmetric
    perturbations and symmetrize the JAX gradient for comparison.
    """
    from serinv.algs.pobtaf_jax import pobtaf_jax_optimized

    n_blocks, block_size, arrow_size = 2, 3, 2
    A_diag, A_lower_diag, A_lower_arrow, A_tip = generate_bta_matrix(
        n_blocks, block_size, arrow_size
    )

    def scalar_output(A_diag, A_lower_diag, A_lower_arrow, A_tip):
        """Scalar function for gradient testing."""
        L_diag, L_lower_diag, L_lower_arrow, L_tip = pobtaf_jax_optimized(
            A_diag, A_lower_diag, A_lower_arrow, A_tip
        )
        return (
            jnp.sum(L_diag ** 2)
            + jnp.sum(L_lower_diag ** 2)
            + jnp.sum(L_lower_arrow ** 2)
            + jnp.sum(L_tip ** 2)
        )

    grad_fn = jax.grad(scalar_output, argnums=(0, 1, 2, 3))
    grads = grad_fn(A_diag, A_lower_diag, A_lower_arrow, A_tip)

    eps = 1e-6

    # Test A_diag (symmetric) - use symmetric perturbations
    fd_grad_diag = np.zeros_like(A_diag)
    for b in range(n_blocks):
        for i in range(block_size):
            for j in range(i, block_size):
                if i == j:
                    A_plus = A_diag.at[b, i, j].add(eps)
                    A_minus = A_diag.at[b, i, j].add(-eps)
                else:
                    A_plus = A_diag.at[b, i, j].add(eps).at[b, j, i].add(eps)
                    A_minus = A_diag.at[b, i, j].add(-eps).at[b, j, i].add(-eps)
                f_plus = scalar_output(A_plus, A_lower_diag, A_lower_arrow, A_tip)
                f_minus = scalar_output(A_minus, A_lower_diag, A_lower_arrow, A_tip)
                fd_grad_diag[b, i, j] = (f_plus - f_minus) / (2 * eps)
                if i != j:
                    fd_grad_diag[b, j, i] = fd_grad_diag[b, i, j]

    # Symmetrize JAX gradient for comparison
    grads_diag_sym = (grads[0] + jnp.swapaxes(grads[0], -1, -2)) / 2
    max_diff = jnp.max(jnp.abs(grads_diag_sym - fd_grad_diag))
    assert max_diff < 1e-6, f"A_diag gradient mismatch: max_diff={max_diff:.2e}"

    # Test A_lower_diag (not symmetric) - direct comparison
    if A_lower_diag.size > 0:
        fd_grad_lower_diag = np.zeros_like(A_lower_diag)
        for idx in np.ndindex(A_lower_diag.shape):
            A_plus = A_lower_diag.at[idx].add(eps)
            A_minus = A_lower_diag.at[idx].add(-eps)
            f_plus = scalar_output(A_diag, A_plus, A_lower_arrow, A_tip)
            f_minus = scalar_output(A_diag, A_minus, A_lower_arrow, A_tip)
            fd_grad_lower_diag[idx] = (f_plus - f_minus) / (2 * eps)

        max_diff = jnp.max(jnp.abs(grads[1] - fd_grad_lower_diag))
        assert max_diff < 1e-6, f"A_lower_diag gradient mismatch: max_diff={max_diff:.2e}"

    # Test A_lower_arrow (not symmetric) - direct comparison
    fd_grad_lower_arrow = np.zeros_like(A_lower_arrow)
    for idx in np.ndindex(A_lower_arrow.shape):
        A_plus = A_lower_arrow.at[idx].add(eps)
        A_minus = A_lower_arrow.at[idx].add(-eps)
        f_plus = scalar_output(A_diag, A_lower_diag, A_plus, A_tip)
        f_minus = scalar_output(A_diag, A_lower_diag, A_minus, A_tip)
        fd_grad_lower_arrow[idx] = (f_plus - f_minus) / (2 * eps)

    max_diff = jnp.max(jnp.abs(grads[2] - fd_grad_lower_arrow))
    assert max_diff < 1e-6, f"A_lower_arrow gradient mismatch: max_diff={max_diff:.2e}"

    # Test A_tip (symmetric) - use symmetric perturbations
    fd_grad_tip = np.zeros_like(A_tip)
    for i in range(arrow_size):
        for j in range(i, arrow_size):
            if i == j:
                A_plus = A_tip.at[i, j].add(eps)
                A_minus = A_tip.at[i, j].add(-eps)
            else:
                A_plus = A_tip.at[i, j].add(eps).at[j, i].add(eps)
                A_minus = A_tip.at[i, j].add(-eps).at[j, i].add(-eps)
            f_plus = scalar_output(A_diag, A_lower_diag, A_lower_arrow, A_plus)
            f_minus = scalar_output(A_diag, A_lower_diag, A_lower_arrow, A_minus)
            fd_grad_tip[i, j] = (f_plus - f_minus) / (2 * eps)
            if i != j:
                fd_grad_tip[j, i] = fd_grad_tip[i, j]

    grads_tip_sym = (grads[3] + grads[3].T) / 2
    max_diff = jnp.max(jnp.abs(grads_tip_sym - fd_grad_tip))
    assert max_diff < 1e-6, f"A_tip gradient mismatch: max_diff={max_diff:.2e}"


def test_gradient_single_block():
    """Test gradient with a single block (edge case)."""
    from serinv.algs.pobtaf_jax import pobtaf_jax_optimized

    n_blocks, block_size, arrow_size = 1, 4, 2
    A_diag, A_lower_diag, A_lower_arrow, A_tip = generate_bta_matrix(
        n_blocks, block_size, arrow_size
    )

    def scalar_output(A_diag, A_lower_diag, A_lower_arrow, A_tip):
        L_diag, L_lower_diag, L_lower_arrow, L_tip = pobtaf_jax_optimized(
            A_diag, A_lower_diag, A_lower_arrow, A_tip
        )
        return jnp.sum(L_diag ** 2) + jnp.sum(L_tip ** 2)

    grad_fn = jax.grad(scalar_output, argnums=(0, 3))
    grads = grad_fn(A_diag, A_lower_diag, A_lower_arrow, A_tip)

    assert grads[0].shape == A_diag.shape
    assert grads[1].shape == A_tip.shape
    assert not jnp.any(jnp.isnan(grads[0]))
    assert not jnp.any(jnp.isnan(grads[1]))


def test_jit_compilation():
    """Test that the function works with JIT compilation."""
    from serinv.algs.pobtaf_jax import pobtaf_jax_optimized

    n_blocks, block_size, arrow_size = 3, 4, 2
    A_diag, A_lower_diag, A_lower_arrow, A_tip = generate_bta_matrix(
        n_blocks, block_size, arrow_size
    )

    @jax.jit
    def jitted_grad(A_diag, A_lower_diag, A_lower_arrow, A_tip):
        def scalar_output(A_diag, A_lower_diag, A_lower_arrow, A_tip):
            L_diag, L_lower_diag, L_lower_arrow, L_tip = pobtaf_jax_optimized(
                A_diag, A_lower_diag, A_lower_arrow, A_tip
            )
            return jnp.sum(L_diag ** 2) + jnp.sum(L_lower_arrow ** 2)

        return jax.grad(scalar_output, argnums=(0, 2))(
            A_diag, A_lower_diag, A_lower_arrow, A_tip
        )

    grads = jitted_grad(A_diag, A_lower_diag, A_lower_arrow, A_tip)
    assert grads[0].shape == A_diag.shape
    assert grads[1].shape == A_lower_arrow.shape


if __name__ == "__main__":
    test_forward_pass_produces_valid_factors()
    test_gradient_with_finite_differences()
    test_gradient_single_block()
    test_jit_compilation()
