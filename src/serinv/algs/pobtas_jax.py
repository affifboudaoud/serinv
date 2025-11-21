
# Copyright 2023-2025 ETH Zurich. All rights reserved.

import jax
from jax import lax


def pobtas_jax_optimized(
    L_diagonal_blocks,
    L_lower_diagonal_blocks,
    L_lower_arrow_blocks,
    L_arrow_tip_block,
    B,
    trans="N",
    partial=False,
):
    """
    JAX-optimized triangular solver using lax.fori_loop.
    """
    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_lower_arrow_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    # Handle n_diag_blocks=1 special case
    if n_diag_blocks == 1:
        if trans == "N":
            if not partial:
                diag_result = jax.scipy.linalg.solve_triangular(
                    L_diagonal_blocks[0],
                    B[:diag_blocksize],
                    lower=True,
                )
                B = B.at[:diag_blocksize].set(diag_result)

                arrow_update = B[-arrow_blocksize:] - (L_lower_arrow_blocks[0] @ diag_result)
                B = B.at[-arrow_blocksize:].set(arrow_update)

                tip_result = jax.scipy.linalg.solve_triangular(
                    L_arrow_tip_block[:], B[-arrow_blocksize:], lower=True
                )
                B = B.at[-arrow_blocksize:].set(tip_result)
        elif trans == "T" or trans == "C":
            if not partial:
                tip_result = jax.scipy.linalg.solve_triangular(
                    L_arrow_tip_block[:],
                    B[-arrow_blocksize:],
                    lower=True,
                    trans="C",
                )
                B = B.at[-arrow_blocksize:].set(tip_result)

                diag_result = jax.scipy.linalg.solve_triangular(
                    L_diagonal_blocks[0],
                    B[:diag_blocksize] - L_lower_arrow_blocks[0].conj().T @ tip_result,
                    lower=True,
                    trans="C",
                )
                B = B.at[:diag_blocksize].set(diag_result)
        else:
            raise ValueError(f"Invalid transpose argument: {trans}.")
        return B

    if trans == "N":
        # Forward substitution using lax.fori_loop
        def forward_body_fn(i, B_state):
            """Loop body for forward substitution."""
            # Extract current block using dynamic_slice
            current_block = lax.dynamic_slice(B_state, (i * diag_blocksize,), (diag_blocksize,))

            # Solve with diagonal block
            result = jax.scipy.linalg.solve_triangular(
                L_diagonal_blocks[i],
                current_block,
                lower=True,
            )

            # Update current block
            B_state = lax.dynamic_update_slice(B_state, result, (i * diag_blocksize,))

            # Extract next block
            next_block = lax.dynamic_slice(B_state, ((i + 1) * diag_blocksize,), (diag_blocksize,))

            # Update next diagonal block
            update = next_block - (L_lower_diagonal_blocks[i] @ result)
            B_state = lax.dynamic_update_slice(B_state, update, ((i + 1) * diag_blocksize,))

            # Update arrow block (always at the end)
            arrow_block = B_state[-arrow_blocksize:]
            arrow_update = arrow_block - (L_lower_arrow_blocks[i] @ result)
            B_state = B_state.at[-arrow_blocksize:].set(arrow_update)

            return B_state

        # Run forward substitution loop
        B = lax.fori_loop(0, n_diag_blocks - 1, forward_body_fn, B)

        if not partial:
            # Handle last diagonal block
            last_diag_result = jax.scipy.linalg.solve_triangular(
                L_diagonal_blocks[n_diag_blocks - 1],
                B[(n_diag_blocks - 1) * diag_blocksize : n_diag_blocks * diag_blocksize],
                lower=True,
            )
            B = B.at[(n_diag_blocks - 1) * diag_blocksize : n_diag_blocks * diag_blocksize].set(last_diag_result)

            # Update arrow block
            last_arrow_update = B[-arrow_blocksize:] - (
                L_lower_arrow_blocks[-1] @ last_diag_result
            )
            B = B.at[-arrow_blocksize:].set(last_arrow_update)

            # Solve arrow tip
            tip_result = jax.scipy.linalg.solve_triangular(
                L_arrow_tip_block[:], B[-arrow_blocksize:], lower=True
            )
            B = B.at[-arrow_blocksize:].set(tip_result)

    elif trans == "T" or trans == "C":
        # Backward substitution
        if not partial:
            # Handle arrow tip first
            tip_result = jax.scipy.linalg.solve_triangular(
                L_arrow_tip_block[:],
                B[-arrow_blocksize:],
                lower=True,
                trans="C",
            )
            B = B.at[-arrow_blocksize:].set(tip_result)

            # Handle last diagonal block
            last_block_result = jax.scipy.linalg.solve_triangular(
                L_diagonal_blocks[-1],
                B[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
                - L_lower_arrow_blocks[-1].conj().T @ tip_result,
                lower=True,
                trans="C",
            )
            B = B.at[-arrow_blocksize - diag_blocksize : -arrow_blocksize].set(last_block_result)

        def backward_body_fn(idx, B_state):
            """Loop body for backward substitution."""
            # Map index for backward iteration: idx=0 -> i=n_diag_blocks-2
            i = n_diag_blocks - 2 - idx

            # Extract blocks using dynamic_slice
            current_block = lax.dynamic_slice(B_state, (i * diag_blocksize,), (diag_blocksize,))
            next_block = lax.dynamic_slice(B_state, ((i + 1) * diag_blocksize,), (diag_blocksize,))
            arrow_block = B_state[-arrow_blocksize:]

            result = jax.scipy.linalg.solve_triangular(
                L_diagonal_blocks[i],
                current_block
                - L_lower_diagonal_blocks[i].conj().T @ next_block
                - L_lower_arrow_blocks[i].conj().T @ arrow_block,
                lower=True,
                trans="C",
            )

            # Update current block
            B_state = lax.dynamic_update_slice(B_state, result, (i * diag_blocksize,))

            return B_state

        # Run backward substitution loop
        B = lax.fori_loop(0, n_diag_blocks - 1, backward_body_fn, B)

    else:
        raise ValueError(f"Invalid transpose argument: {trans}.")

    return B
