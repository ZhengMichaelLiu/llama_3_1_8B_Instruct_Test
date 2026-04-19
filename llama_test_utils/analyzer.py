"""
Q/K and Attention Matrix Analyzer

Analyzes and visualizes Q, K, Pre-Softmax Attention, and Post-Softmax Attention matrices.
Organizes plots in a layer/head hierarchy.

Author: Zheng Liu
Date: 2025/07/05
"""

import os
import sys

import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix' # Often used for math in papers
plt.rcParams['axes.linewidth'] = 1.5 # Thicker plot edges globally
plt.rcParams['axes.edgecolor'] = 'yellow' # Darker axes for better print visibility

def get_device():
    """Determine the best available device: CUDA, MPS, or CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

def find_blocks_chunked(
    input_tensor, current_index, threshold, num_to_choose, decoding: bool, mode: str = "both", causal=True
):
    """
        Finds and selects relevant blocks of attention for transformer-based models based on a
        threshold or a predefined number of blocks.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
        - current_index (int): The current index in the sequence processing.
        - threshold (float or None): A threshold value used to determine the minimum attention weight sum.
        - num_to_choose (int or None): The number of blocks to be selected, ensuring sufficient information retrieval.
        - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
        - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
        - causal (bool): If True, applies causal masking to prevent future information leakage.

        Returns:
        - torch.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
        indicating which blocks should be attended to.
    """
    device = input_tensor.device

    assert threshold is None or num_to_choose is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    # 0 -- -- -- -- current_index
    # 0 -- -- -- -- -- current_index+1
    # 0 -- -- -- -- -- ----------- current_index + chunk_num - 1
    if mode == "prefill" and decoding:
        return torch.ones_like(input_tensor, dtype=torch.bool)
    if mode == "decode" and not decoding:
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(1, head_num, chunk_num, chunk_num, device=device)
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(input_tensor, dtype=torch.bool)[:, :, 0 : current_index + 1],
                    torch.zeros_like(input_tensor, dtype=torch.bool)[:, :, current_index + 1 :],
                ],
                dim=-1,
            )
        else:
            return mask
    input_tensor = input_tensor.to(torch.float32)

    if threshold is not None:
        total_sum = input_tensor.sum(dim=-1, keepdim=True)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(torch.float32).to(device)
            required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1
            ).expand((batch_size, head_num, chunk_num, 1)).to(device)
        else:
            required_sum = total_sum * threshold
        if causal:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            mask[:, :, :, 0] = 1
            mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            other_values = input_tensor.masked_fill(
                mask, 0
            )
            sorted_values, _ = torch.sort(
                other_values, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)

            sorted_values = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=device
                    ),
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )

            _, index = torch.sort(
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask,index,0)
            mask = mask.view(batch_size,head_num*chunk_num,block_num)
            index = index.view(batch_size,head_num*chunk_num,block_num)
            mask[:,torch.arange(mask.shape[1], device=device).unsqueeze(dim=-1),index] = True
            mask = mask.view(batch_size,head_num,chunk_num,block_num)
            # assert(bool((torch.where(mask,input_tensor,0).sum(dim=-1,keepdim=True) >= required_sum*0.99).all()))
        else:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(
                input_tensor, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(device)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        raise NotImplementedError("block num chunk prefill not impleted")

    try:
        if causal:
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except:
        mask[:, :, :, current_index + chunk_num :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(input_tensor,dtype=bool,device=device)
            lambda_mask[:,:,:,0] = 1
            lambda_mask[:,:,:,current_index:current_index+chunk_num] = torch.eye(chunk_num, device=device).unsqueeze(0).unsqueeze(0).expand(1,head_num,chunk_num,chunk_num)
            assert(torch.where(lambda_mask,mask,True).all())

    return mask

def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=False,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    device = get_device()
    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    query_states = query_states.to(device)
    key_states = key_states.to(device)

    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to(device)
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(device)
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        chunked_query = reshaped_query[
            :,
            :,
            (chunk_idx * reshaped_chunk_size)
            // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
            // kdb,
            :,
        ]
        attn_weights_slice = torch.matmul(
            chunked_query,
            reshaped_key.transpose(2, 3),
        ).to(device)

        attn_weights_slice = (
            attn_weights_slice / math.sqrt(head_dim) / stride / norm
        )
        if causal:
            causal_mask = torch.zeros(
                (
                    batch_size,
                    num_q_head,
                    reshaped_chunk_size,
                    reshaped_chunk_size * k_chunk_num,
                ),
                device=device,
            )

            # Fix 1: Only mask padding if there is actually padding (avoid [-0:] selecting all)
            if k_reshaped_num_to_pad > 0:
                causal_mask[:, :, :, -k_reshaped_num_to_pad:] = float("-inf")

            chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
            chunk_end = chunk_start + reshaped_chunk_size

            # Fix 2: Safer triu generation for MPS/Mac compatibility
            # Create 1s and 0s first, then fill -inf
            local_mask = torch.ones(
                1,
                num_q_head,
                reshaped_chunk_size,
                reshaped_chunk_size,
                device=device,
            )
            local_mask = torch.triu(local_mask, diagonal=1)
            # Upper triangle becomes -inf, lower remains 0
            causal_mask[:, :, :, chunk_start:chunk_end] = local_mask.masked_fill(local_mask == 1, float("-inf"))

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                # Fix 3: Safety check for Query padding to avoid [-0:] slicing
                rows_to_mask = q_reshaped_num_to_pad // kdb
                if rows_to_mask > 0:
                    causal_mask[:, :, -rows_to_mask:, :] = float("-inf")

            causal_mask[:, :, :, chunk_end:] = float("-inf")
            causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
            attn_weights_slice = attn_weights_slice + causal_mask.to(device)

        if softmax:
            attn_weights_slice = F.softmax(
                attn_weights_slice, dim=-1, dtype=torch.float32
            ).to(pad_query_states.dtype)
        else:
            attn_weights_slice = torch.exp(attn_weights_slice).to(pad_query_states.dtype)

        attn_weights_slice = torch.nan_to_num(attn_weights_slice, nan=0.0)

        attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

        if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
            attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

        attn_sum = (
            attn_weights_slice.view(
                batch_size,
                num_kv_head,
                num_blocks_per_chunk,
                reshaped_block_size // kdb,
                -1,
                reshaped_block_size,
            )
            .sum(dim=-1)
            .sum(dim=-2)
            .to(device)
        )
        del chunked_query

        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool, device=device
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    return attn_sums, simple_masks

def create_block_mask(
    q, k,
    block_size=128,
    patch_size=8, # Can be 8, 16, 32...
    causal=True,
    cumulative_p_threshold=0.9,
    chunk_size=4096,
    keep_sink=True,
    keep_local=True,
):
    """
    Create block sparse mask using double-band anti-diagonal pattern.

    Pattern for patch_size=8 (K=2, each row selects 2 consecutive K positions):
       0 1 2 3 4 5 6 7
      +----------------
    0 | . . . . . . 1 1
    1 | . . . . 1 1 . .
    2 | . . 1 1 . . . .
    3 | 1 1 . . . . . .
    4 | . . . . . . 1 1
    5 | . . . . 1 1 . .
    6 | . . 1 1 . . . .
    7 | 1 1 . . . . . .

    Row i selects columns: (patch_size - 2 - 2*i) % patch_size and (patch_size - 1 - 2*i) % patch_size
    """
    device = q.device
    B, H, Q_Len, D = q.shape
    _, _, K_Len, _ = k.shape

    # --- GENERALIZED CONSTANTS ---
    # The pattern always selects 2 columns (e.g., "1 1")
    K_BAND = 2

    # The pattern always repeats twice vertically (Top Half / Bottom Half)
    # e.g., for P=16, rows 0-7 are unique, rows 8-15 repeat them.
    Q_REPEATS = 2

    # Derived dimensions
    # For P=16: 16 // 2 = 8 groups.
    num_groups = patch_size // K_BAND

    # 1. Padding
    q_pad_len = ((Q_Len + block_size - 1) // block_size) * block_size
    if q_pad_len > Q_Len: q = F.pad(q, (0, 0, 0, q_pad_len - Q_Len))
    k_pad_len = ((K_Len + block_size - 1) // block_size) * block_size
    if k_pad_len > K_Len: k = F.pad(k, (0, 0, 0, k_pad_len - K_Len))

    num_chunks = (q_pad_len + chunk_size - 1) // chunk_size
    num_k_patches = k_pad_len // patch_size
    k_block_num = k_pad_len // block_size

    # =================================================================================
    # PHASE 1: K Projection (Sum & Accumulate)
    # =================================================================================
    # Goal: Sum consecutive K_BAND columns.
    # Shape: (B, H, K_patches, num_groups, K_BAND, D)
    k_patched = k.view(B, H, num_k_patches, num_groups, K_BAND, D)

    # Sum the band (0+1, 2+3, ...)
    # Shape: (B, H, K_patches, num_groups, D)
    k_summed = k_patched.sum(dim=-2)

    # ALIGNMENT:
    # Pattern Logic: Row 0 (Index 0) targets the LAST K-Group (Index N).
    # We FLIP K here so that Index 0 contains the Last Group's data.
    # This aligns the indices: Q[0] will multiply with K_flipped[0] (which is G_last).
    k_aligned = k_summed.flip(dims=[-2])

    # Flatten for MatMul
    # Shape: (B, H, num_groups * D, K_patches) (Transposed)
    k_in = k_aligned.reshape(B, H, num_k_patches, -1).transpose(-1, -2)

    block_mask_list = []
    attn_probs_list = []

    for i in range(num_chunks):
        start_token = i * chunk_size
        end_token = min(start_token + chunk_size, q_pad_len)
        chunk_len = end_token - start_token
        num_q_patches = chunk_len // patch_size

        # =============================================================================
        # PHASE 2: Q Projection (Sum & Accumulate)
        # =============================================================================
        q_chunk = q[:, :, start_token:end_token, :]

        # Goal: Sum the repeating halves.
        # Structure: The patch is split into Q_REPEATS (2) equal vertical parts.
        # Height of one part = patch_size // Q_REPEATS
        rows_per_part = patch_size // Q_REPEATS

        # View: (B, H, Q_patches, Q_REPEATS, rows_per_part, D)
        # Dim -3 (Q_REPEATS) separates the Top Half from Bottom Half
        q_view = q_chunk.view(B, H, num_q_patches, Q_REPEATS, rows_per_part, D)

        # Sum the halves (Row i + Row i+P/2)
        # Shape: (B, H, Q_patches, rows_per_part, D)
        # Note: rows_per_part is exactly equal to num_groups (P/2 == P/2)
        q_summed = q_view.sum(dim=-3)

        # Flatten for MatMul
        # Shape: (B, H, Q_patches, num_groups * D)
        q_in = q_summed.reshape(B, H, num_q_patches, -1)

        # =============================================================================
        # PHASE 3: Efficient Scoring
        # =============================================================================
        # MatMul calculates: (Sum of Q Halves) . (Sum of K Bands)
        scores = torch.matmul(q_in, k_in)

        scores.div_(math.sqrt(D) * patch_size * 2)

        if causal:
            q_idx = torch.arange(num_q_patches, device=device).view(1, -1, 1) + (start_token // patch_size)
            k_idx = torch.arange(num_k_patches, device=device).view(1, 1, -1)
            mask = q_idx >= k_idx
            scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))

        # =============================================================================
        # PHASE 4: Block Aggregation
        # =============================================================================
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs.nan_to_num_(0.0)

        patches_per_block = block_size // patch_size
        num_q_chunk_blocks = chunk_len // block_size

        # Reshape to Blocks
        attn_probs = attn_probs.view(B, H, num_q_chunk_blocks, patches_per_block, k_block_num, patches_per_block)

        # Aggregation: Sum K, Sum Q (Total Mass logic)
        block_attn_probs = attn_probs.sum(dim=-1).sum(dim=-2)
        attn_probs_list.append(block_attn_probs)

        # =============================================================================
        # PHASE 5: Selection Logic (Standard)
        # =============================================================================
        chunk_start_block = start_token // block_size
        q_blk_idx = torch.arange(num_q_chunk_blocks, device=device).view(-1, 1) + chunk_start_block
        k_blk_idx = torch.arange(k_block_num, device=device).view(1, -1)

        mandatory_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        if keep_sink: mandatory_mask[..., 0] = True
        if keep_local: mandatory_mask |= (q_blk_idx == k_blk_idx).unsqueeze(0).unsqueeze(0)

        sort_scores = block_attn_probs.clone()
        sort_scores.masked_fill_(mandatory_mask, 1000.0)

        _, sorted_indices = torch.sort(sort_scores, dim=-1, descending=True)
        sorted_probs = torch.gather(block_attn_probs, -1, sorted_indices)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        total_mass = block_attn_probs.sum(dim=-1, keepdim=True)
        required_mass = total_mass * cumulative_p_threshold
        cumsum_without_self = torch.cat([torch.zeros_like(cumsum_probs[..., :1]), cumsum_probs[..., :-1]], dim=-1)
        keep_mask_sorted = cumsum_without_self < required_mass

        chunk_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        chunk_mask.scatter_(dim=-1, index=sorted_indices, src=keep_mask_sorted)
        chunk_mask |= mandatory_mask

        if causal:
            chunk_mask &= (q_blk_idx >= k_blk_idx).unsqueeze(0).unsqueeze(0)

        block_mask_list.append(chunk_mask)

    return torch.cat(block_mask_list, dim=-2), torch.cat(attn_probs_list, dim=-2)

class QKAAnalyzer:
    """Analyzes Q, K, and Attention matrices collected from the model."""

    def __init__(self):
        pass

    #################################################
    def visualize_qk_matrix(self, q_matrices: Dict[int, torch.Tensor], k_matrices: Dict[int, torch.Tensor], output_dir: str = "qk_matrix_visualization"):
        """
        Visualizes Q and K matrices side-by-side for each layer and head.
        """
        self.base_output_dir = output_dir
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        layers = sorted(q_matrices.keys())
        print(f"Starting Q/K matrix visualization for {len(layers)} layers...")

        for layer_idx in layers:
            layer_dir = os.path.join(self.base_output_dir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            Q_layer = q_matrices[layer_idx] # [batch, heads, seq, dim]
            K_layer = k_matrices[layer_idx]
            num_heads = Q_layer.shape[1]

            for head_idx in range(num_heads):
                q_head = Q_layer[0, head_idx].detach().cpu().numpy()
                k_head = K_layer[0, head_idx].detach().cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f"Layer {layer_idx} - Head {head_idx} Q/K Matrices", fontsize=16)

                # Plot Q
                im_q = axes[0].imshow(q_head, aspect='auto', cmap='viridis', interpolation='nearest')
                axes[0].set_title("Query Matrix")
                axes[0].set_xlabel("Head Dimension")
                axes[0].set_ylabel("Sequence Position")
                fig.colorbar(im_q, ax=axes[0])

                # Plot K
                im_k = axes[1].imshow(k_head, aspect='auto', cmap='viridis', interpolation='nearest')
                axes[1].set_title("Key Matrix")
                axes[1].set_xlabel("Head Dimension")
                axes[1].set_ylabel("Sequence Position")
                fig.colorbar(im_k, ax=axes[1])

                plt.tight_layout()
                save_path = os.path.join(layer_dir, f"head_{head_idx:02d}.png")
                plt.savefig(save_path, dpi=100)
                plt.close(fig)

    #################################################
    def visualize_high_score_parts_in_post_softmax_matrix(self, attention_matrices: Dict[int, torch.Tensor], target_mass: float = 0.9, output_dir: str = "high_score_in_post_softmax_results"):
        """
        Generate visualizations for all layers and heads.

        Args:
            attention_matrices: Dictionary of Attention matrices per layer.
            output_dir: Directory to save the analysis results.
        """
        self.base_output_dir = output_dir
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        layers = sorted(attention_matrices.keys())
        print(f"Starting visualization for {len(layers)} layers...")
        print(f"Output directory: {self.base_output_dir}")

        for layer_idx in layers:
            self._visualize_layer(
                layer_idx,
                attention_matrices.get(layer_idx),
                target_mass=target_mass
            )

    def _visualize_layer(self, layer_idx: int, Attn: Optional[torch.Tensor], target_mass: float = 0.9):
        """Visualize all heads in a specific layer."""
        # Create layer directory: qka_analysis/layer_xx
        layer_dir = os.path.join(self.base_output_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        num_heads = Attn.shape[1]
        print(f"Visualizing Layer {layer_idx} ({num_heads} heads)...")

        for head_idx in range(num_heads):
            self._visualize_head(layer_idx, head_idx, Attn, target_mass, layer_dir)

    def _visualize_head(self, layer_idx: int, head_idx: int, Attn: Optional[torch.Tensor], target: float, output_dir: str):
        """Generate plot for a single head: Post-Softmax High Scores."""

        post_softmax_np = None
        sparsity_info = "N/A"

        if Attn is not None:
            # Collect original attention matrix [seq_len, seq_len]
            attn_tensor = Attn[0, head_idx]
            seq_len = attn_tensor.shape[0]

            # --- Core Logic: Compute Top-P (Cumulative Probability) ---
            target_mass = target  # We want to know how many tokens constitute target% of the weight

            # 1. Sort each row in descending order
            sorted_vals, sorted_indices = torch.sort(attn_tensor, descending=True, dim=-1)

            # 2. Compute cumulative sum
            cumsum_vals = torch.cumsum(sorted_vals, dim=-1)

            # 3. Create mask
            cumsum_without_self = torch.cat([torch.zeros_like(cumsum_vals[..., :1]), cumsum_vals[..., :-1]], dim=-1)
            mask_remove = cumsum_without_self >= target_mass
            mask_remove[..., 0] = 0 # Always keep the largest value

            # 4. Statistics
            # Count how many tokens are kept in each row
            kept_counts = (~mask_remove).sum(dim=-1).float()
            avg_kept = kept_counts.mean().item()

            # Compute sparsity relative to the Causal Context (denominator is the number of tokens visible in the current row, not the total seq_len)
            # Row i can see i+1 tokens
            causal_lengths = torch.arange(1, seq_len + 1, device=attn_tensor.device).float()
            sparsity_ratios = kept_counts / causal_lengths
            avg_ratio = sparsity_ratios.mean().item() * 100

            sparsity_info = f"Top-{int(target_mass*100)}%: Avg {avg_ratio:.1f}% of context needed"

            # 5. Construct sparse matrix for visualization
            keep_mask_original = torch.zeros_like(attn_tensor, dtype=torch.bool)
            keep_mask_original.scatter_(dim=-1, index=sorted_indices, src=~mask_remove)

            binary_vis = torch.full_like(attn_tensor, float('nan'))

            # Set all selected positions to 1.0 (uniform red)
            binary_vis[keep_mask_original] = 1.0

            post_softmax_np = binary_vis.detach().cpu().numpy()

        # Setup Plot
        # Smaller figure size for paper, larger fonts
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(f"Layer {layer_idx} - Head {head_idx}", fontsize=18, fontweight='bold')

        # Post-Softmax Attention
        if post_softmax_np is not None:
            # Revert to imshow to avoid over-dense scatter dots, showing the true pattern
            im = ax.imshow(post_softmax_np, aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1.0)

            # Simplified title, larger fonts for paper readability
            ax.set_title(sparsity_info, fontsize=16)
            ax.set_xlabel("Key Position", fontsize=18, fontweight='bold')
            ax.set_ylabel("Query Position", fontsize=18, fontweight='bold')

            # Increase tick label size and tick thickness for clarity
            ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)

            # Make the edges of plots more salient and clear
            for spine in ax.spines.values():
                spine.set_linewidth(4.0)
                spine.set_color('#FFC000') # Thicker yellow edge

        else:
            ax.text(0.5, 0.5, "Not Available", ha='center', va='center')

        plt.tight_layout()

        # Save: qka_analysis/layer_xx/head_xx.pdf
        filename = f"head_{head_idx:02d}.pdf"
        save_path = os.path.join(output_dir, filename)

        # Modified: Save as PDF with tight bounding box for LaTeX inclusion
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

    #################################################
    def visualize_layer_head_sparsity_heatmap(self, attention_matrices: Dict[int, torch.Tensor], target_mass: float = 0.95, output_dir: str = "layer_head_sparsity_heatmap"):
        """
        Visualizes the sparsity of all layers and heads as a heatmap.
        """
        self.base_output_dir = output_dir
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        layers = sorted(attention_matrices.keys())
        if not layers:
            return

        num_layers = len(layers)
        num_heads = attention_matrices[layers[0]].shape[1]

        sparsity_matrix = np.zeros((num_layers, num_heads))

        print(f"Calculating sparsity heatmap for {num_layers} layers and {num_heads} heads...")

        for idx, layer_idx in enumerate(layers):
            attn_layer = attention_matrices.get(layer_idx)
            if attn_layer is not None:
                for head_idx in range(num_heads):
                    attn_head = attn_layer[0, head_idx]
                    ratio = self._calculate_sparsity_ratio(attn_head, target_mass)
                    # ratio is returned as a percentage 0-100, scale it to 0-1.0
                    sparsity_matrix[idx, head_idx] = ratio / 100.0

        fig, ax = plt.subplots(figsize=(6, 5))

        try:
            import seaborn as sns
            cmap = sns.color_palette("Blues", as_cmap=True)
        except ImportError:
            cmap = 'Blues' if 'Blues' in plt.colormaps() else 'viridis'

        im = ax.imshow(sparsity_matrix, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=1.0)

        # Calculate overall mean sparsity percentage for the title
        overall_mean_sparsity = np.mean(sparsity_matrix) * 100

        ax.set_title(f"Density of Attention\nTarget Mass={target_mass*100:.0f}%, Avg={overall_mean_sparsity:.1f}%", fontsize=18)
        ax.set_xlabel("Head", fontsize=18, fontweight='bold')
        ax.set_ylabel("Layer", fontsize=18, fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=14)

        # Adjust ticks to match the paper style (1, 4, 8, 12, ...)
        x_ticks = [0, 3, 7, 11, 15, 19, 23, 27, 31]
        y_ticks = [0, 3, 7, 11, 15, 19, 23, 27, 31]

        # Filter out ticks that exceed our actual dimensions
        x_ticks = [t for t in x_ticks if t < num_heads]
        y_ticks = [t for t in y_ticks if t < num_layers]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([t + 1 for t in x_ticks])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([t + 1 for t in y_ticks])

        # Style the edges
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        save_path = os.path.join(self.base_output_dir, "layer_head_sparsity_heatmap.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Layer-Head Sparsity Heatmap saved to {save_path}")

    #################################################
    def visualize_sparsity_vs_mass(self, attention_matrices: Dict[int, torch.Tensor], mass_levels: list = [0.3, 0.5, 0.8, 0.9, 0.95, 0.99], output_dir: str = "cumulative_percentage_sparsity_results"):
        """
        Generates a bar plot showing the average percentage of key values needed
        to achieve different levels of cumulative probability mass.
        """
        self.base_output_dir = output_dir
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        print(f"Starting sparsity analysis for mass levels: {mass_levels}...")

        results = [] # List of (mass, avg_ratio, std_ratio)

        for mass in mass_levels:
            all_head_ratios = []

            for layer_idx, attn_layer in attention_matrices.items():
                if attn_layer is None: continue

                # attn_layer shape: [batch, num_heads, seq_len, seq_len]
                num_heads = attn_layer.shape[1]
                for head_idx in range(num_heads):
                    attn_head = attn_layer[0, head_idx] # [seq_len, seq_len]

                    # Calculate ratio for this head
                    ratio = self._calculate_sparsity_ratio(attn_head, mass)
                    all_head_ratios.append(ratio)

            # Aggregate for this mass level
            if all_head_ratios:
                avg_ratio = np.mean(all_head_ratios)
                std_ratio = np.std(all_head_ratios)
                results.append((mass, avg_ratio, std_ratio))
                print(f"Mass {mass}: Avg {avg_ratio:.2f}% keys needed")

        # Plotting
        self._plot_sparsity_bar_chart(results)

    def _calculate_sparsity_ratio(self, attn_tensor: torch.Tensor, target_mass: float) -> float:
        """Calculates the average percentage of context needed for a single head."""
        seq_len = attn_tensor.shape[0]

        # 1. Sort descending
        sorted_vals, _ = torch.sort(attn_tensor, descending=True, dim=-1)

        # 2. Cumsum
        cumsum_vals = torch.cumsum(sorted_vals, dim=-1)

        # 3. Mask
        mask_remove = cumsum_vals > target_mass
        mask_remove[..., 1:] = mask_remove[..., :-1].clone()
        mask_remove[..., 0] = 0 # Always keep at least one

        # 4. Count kept
        kept_counts = (~mask_remove).sum(dim=-1).float()

        # 5. Causal lengths (denominator)
        # Row i can attend to i+1 tokens (0..i)
        causal_lengths = torch.arange(1, seq_len + 1, device=attn_tensor.device).float()

        # 6. Ratios
        sparsity_ratios = kept_counts / causal_lengths

        # Return average percentage (0-100)
        return sparsity_ratios.mean().item() * 100

    def _plot_sparsity_bar_chart(self, results):
        if not results:
            print("No results to plot.")
            return

        masses = [r[0] for r in results]
        means = [r[1] for r in results]
        stds = [r[2] for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(masses))

        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', align='center')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{m}" for m in masses])
        ax.set_xlabel('Cumulative Probability Mass Target')
        ax.set_ylabel('Avg % of Keys Needed (Context)')
        ax.set_title('Attention Sparsity: % of Keys Needed vs. Cumulative Probability')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        plt.tight_layout()
        save_path = os.path.join(self.base_output_dir, "sparsity_vs_mass.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"Sparsity plot saved to {save_path}")

    #################################################
    def visualize_block_similarity(self, q_matrices: Dict[int, torch.Tensor], k_matrices: Dict[int, torch.Tensor], block_sizes: list = [16, 32, 64], output_dir: str = "block_similarity_analysis"):
        """
        Analyzes distance and angular variance within blocks of Q and K matrices.
        """
        self.base_output_dir = output_dir
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        layers = sorted(q_matrices.keys())
        print(f"Starting Block Similarity Analysis for {len(layers)} layers...")

        for layer_idx in layers:
            layer_dir = os.path.join(self.base_output_dir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            Q_layer = q_matrices[layer_idx]
            K_layer = k_matrices[layer_idx]
            num_heads = Q_layer.shape[1]

            for head_idx in range(num_heads):
                q_head = Q_layer[0, head_idx] # [seq, dim]
                k_head = K_layer[0, head_idx]

                # Calculate metrics for different block sizes
                q_metrics = [self._calculate_block_metrics(q_head, bs) for bs in block_sizes]
                k_metrics = [self._calculate_block_metrics(k_head, bs) for bs in block_sizes]

                # Unzip metrics: [(dist, ang), ...] -> [dist, ...], [ang, ...]
                q_dist_vars = [m[0] for m in q_metrics]
                q_ang_vars = [m[1] for m in q_metrics]
                k_dist_vars = [m[0] for m in k_metrics]
                k_ang_vars = [m[1] for m in k_metrics]

                # Plotting
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"Layer {layer_idx} - Head {head_idx} Block Similarity Variance", fontsize=16)

                x_pos = np.arange(len(block_sizes))
                width = 0.6

                # Q Distance
                axes[0, 0].bar(x_pos, q_dist_vars, width, color='skyblue')
                axes[0, 0].set_title("Q Matrix: Block Distance Variance")
                axes[0, 0].set_xticks(x_pos)
                axes[0, 0].set_xticklabels(block_sizes)
                axes[0, 0].set_ylabel("Variance")

                # K Distance
                axes[0, 1].bar(x_pos, k_dist_vars, width, color='salmon')
                axes[0, 1].set_title("K Matrix: Block Distance Variance")
                axes[0, 1].set_xticks(x_pos)
                axes[0, 1].set_xticklabels(block_sizes)

                # Q Angular
                axes[1, 0].bar(x_pos, q_ang_vars, width, color='skyblue')
                axes[1, 0].set_title("Q Matrix: Block Angular Variance (Radians)")
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(block_sizes)
                axes[1, 0].set_xlabel("Block Size")
                axes[1, 0].set_ylabel("Variance")

                # K Angular
                axes[1, 1].bar(x_pos, k_ang_vars, width, color='salmon')
                axes[1, 1].set_title("K Matrix: Block Angular Variance (Radians)")
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(block_sizes)
                axes[1, 1].set_xlabel("Block Size")

                plt.tight_layout()
                save_path = os.path.join(layer_dir, f"head_{head_idx:02d}.png")
                plt.savefig(save_path, dpi=100)
                plt.close(fig)

    def _calculate_block_metrics(self, tensor: torch.Tensor, block_size: int):
        """
        Calculates average variance of pairwise distances and angles within blocks.
        Returns: (avg_dist_variance, avg_ang_variance)
        """
        # Ensure tensor is float32 for cdist compatibility
        tensor = tensor.float()

        seq_len, dim = tensor.shape
        num_blocks = seq_len // block_size
        if num_blocks == 0:
            return 0.0, 0.0

        # Truncate to fit blocks
        data = tensor[:num_blocks*block_size].view(num_blocks, block_size, dim)

        dist_vars = []
        ang_vars = []

        # Pre-calculate indices for upper triangle (excluding diagonal)
        triu_idx = torch.triu_indices(block_size, block_size, offset=1, device=tensor.device)

        for i in range(num_blocks):
            block = data[i] # [block_size, dim]

            # --- Distance Variance ---
            # Pairwise Euclidean distance
            dists = torch.cdist(block, block, p=2) # [block_size, block_size]
            valid_dists = dists[triu_idx[0], triu_idx[1]]
            if len(valid_dists) > 0:
                dist_vars.append(torch.var(valid_dists).item())

            # --- Angular Variance ---
            # Cosine similarity -> Angle
            norms = torch.norm(block, p=2, dim=1, keepdim=True)
            normalized_block = block / (norms + 1e-8)
            cos_sim = torch.mm(normalized_block, normalized_block.t())
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            angles = torch.acos(cos_sim)

            valid_angles = angles[triu_idx[0], triu_idx[1]]
            if len(valid_angles) > 0:
                ang_vars.append(torch.var(valid_angles).item())

        avg_dist_var = np.mean(dist_vars) if dist_vars else 0.0
        avg_ang_var = np.mean(ang_vars) if ang_vars else 0.0

        return avg_dist_var, avg_ang_var

    #################################################
    def visualize_grid_attention_block_sparse_mask(self, q_matrices: Dict[int, torch.Tensor], k_matrices: Dict[int, torch.Tensor], attention_matrices: Dict[int, torch.Tensor], target_mass: float = 0.9, block_size: int = 128, grid_size: int = 8, output_dir: str = "grid_attention_block_sparse_mask_results"):
        """
        Visualizes comparison between Post-Softmax High Scores, Grid Attention Block Sparse Mask, and XAttention Mask.
        """
        self.base_output_dir = output_dir
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        layers = sorted(q_matrices.keys())
        print(f"Starting Grid Attention & XAttention Mask Visualization for {len(layers)} layers...")

        for layer_idx in layers:
            layer_dir = os.path.join(self.base_output_dir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            Q_layer = q_matrices[layer_idx] # [batch, heads, seq, dim]
            K_layer = k_matrices[layer_idx]
            Attn_layer = attention_matrices.get(layer_idx)

            if Attn_layer is None:
                continue

            num_heads = Q_layer.shape[1]

            for head_idx in range(num_heads):
                # 1. Prepare Data
                # Extract single head data for visualization logic
                attn_head = Attn_layer[0, head_idx] # [seq, seq]
                seq_len = attn_head.shape[0]

                # 2. Left Plot: Post-Softmax High Score (Ground Truth)
                sorted_vals, sorted_indices = torch.sort(attn_head, descending=True, dim=-1)
                cumsum_vals = torch.cumsum(sorted_vals, dim=-1)

                # Modified: Use cumsum_without_self to strictly include the position that causes the sum to exceed the threshold.
                cumsum_without_self = torch.cat([torch.zeros_like(cumsum_vals[..., :1]), cumsum_vals[..., :-1]], dim=-1)
                mask_remove = cumsum_without_self >= target_mass
                mask_remove[..., 0] = 0

                keep_mask_original = torch.zeros_like(attn_head, dtype=torch.bool)
                keep_mask_original.scatter_(dim=-1, index=sorted_indices, src=~mask_remove)

                post_softmax_vis = torch.full_like(attn_head, float('nan'))
                post_softmax_vis[keep_mask_original] = 1.0

                # Prepare inputs for mask generation
                q_input = Q_layer[:, head_idx:head_idx+1, :, :]
                k_input = K_layer[:, head_idx:head_idx+1, :, :]

                # 3. Middle Plot: Grid Attention Block Mask (Prediction)
                block_mask, attn_probs = create_block_mask(
                    q_input, # (Batch, Num_heads, Q_Len, D)
                    k_input, # (Batch, Num_heads, K_Len, D)
                    block_size=128,
                    patch_size=grid_size,
                    causal=True,
                    cumulative_p_threshold=target_mass,
                    chunk_size=4096,
                    keep_sink=True,
                    keep_local=True,
                )

                block_mask_2d = block_mask[0, 0]
                block_mask_np = block_mask_2d.detach().cpu().numpy().astype(float)

                attn_probs_2d = attn_probs[0, 0]
                attn_probs_np = attn_probs_2d.detach().cpu().numpy()

                # 4. Right Plot: XAttention Block Mask (Comparison)
                xattn_mask_np = None
                xattn_probs_np = None # Initialize variable for scores
                xattn_score, xattn_mask = xattn_estimate(
                    q_input,
                    k_input,
                    block_size=block_size,
                    stride=grid_size,
                    threshold=target_mass,
                    keep_sink=True,
                    keep_recent=True,
                    use_triton=False # Disable triton for analysis
                )
                xattn_mask_2d = xattn_mask[0, 0]
                xattn_mask_np = xattn_mask_2d.detach().cpu().numpy().astype(float)

                xattn_score_2d = xattn_score[0, 0]
                xattn_probs_np = xattn_score_2d.detach().cpu().numpy()

                # 5. Plotting
                fig, axes = plt.subplots(1, 3, figsize=(24, 7))
                fig.suptitle(f"Layer {layer_idx} - Head {head_idx}: Comparison (Target Mass {target_mass})", fontsize=16)

                # Plot 1: Ground Truth
                im1 = axes[0].imshow(post_softmax_vis, aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1.0)
                axes[0].set_title("Post-Softmax High Scores (Ground Truth)")
                axes[0].set_xlabel("Key Position")
                axes[0].set_ylabel("Query Position")

                # Plot 2: Grid Attention
                if block_mask_np is not None:
                    n_q_blocks, n_k_blocks = block_mask_np.shape
                    extent = [0, n_k_blocks * block_size, n_q_blocks * block_size, 0]

                    im2 = axes[1].imshow(block_mask_np, aspect='auto', cmap='Blues', interpolation='nearest',
                                       extent=extent, vmin=0, vmax=1.0)
                    axes[1].set_title(f"Grid Attention (Block {block_size})")
                    axes[1].set_xlabel("Key Position (approx)")
                    axes[1].set_xlim(0, seq_len)
                    axes[1].set_ylim(seq_len, 0)

                    # Add score annotations
                    for r in range(n_q_blocks):
                        for c in range(n_k_blocks):
                            x_pos = (c + 0.5) * block_size
                            y_pos = (r + 0.5) * block_size
                            if x_pos > seq_len or y_pos > seq_len: continue

                            is_selected = block_mask_np[r, c] > 0.5
                            prob = attn_probs_np[r, c]
                            text_str = f"{prob:.2f}"
                            text_color = 'white' if is_selected else 'black'
                            font_weight = 'bold' if is_selected else 'normal'

                            axes[1].text(x_pos, y_pos, text_str, ha='center', va='center',
                                       color=text_color, fontsize=5, fontweight=font_weight)
                else:
                    axes[1].text(0.5, 0.5, "Grid Attn Failed", ha='center')

                # Plot 3: XAttention
                if xattn_mask_np is not None:
                    n_q_blocks_x, n_k_blocks_x = xattn_mask_np.shape
                    extent_x = [0, n_k_blocks_x * block_size, n_q_blocks_x * block_size, 0]

                    im3 = axes[2].imshow(xattn_mask_np, aspect='auto', cmap='Greens', interpolation='nearest',
                                       extent=extent_x, vmin=0, vmax=1.0)
                    axes[2].set_title(f"XAttention (Stride {grid_size})")
                    axes[2].set_xlabel("Key Position (approx)")
                    axes[2].set_xlim(0, seq_len)
                    axes[2].set_ylim(seq_len, 0)

                    # Add score annotations for XAttention
                    if xattn_probs_np is not None:
                        for r in range(n_q_blocks_x):
                            for c in range(n_k_blocks_x):
                                x_pos = (c + 0.5) * block_size
                                y_pos = (r + 0.5) * block_size
                                if x_pos > seq_len or y_pos > seq_len: continue

                                is_selected = xattn_mask_np[r, c] > 0.5
                                prob = xattn_probs_np[r, c]
                                text_str = f"{prob:.2f}"
                                text_color = 'white' if is_selected else 'black'
                                font_weight = 'bold' if is_selected else 'normal'

                                axes[2].text(x_pos, y_pos, text_str, ha='center', va='center',
                                           color=text_color, fontsize=5, fontweight=font_weight)
                else:
                    axes[2].text(0.5, 0.5, "XAttention Failed", ha='center')

                plt.tight_layout()
                save_path = os.path.join(layer_dir, f"head_{head_idx:02d}.png")
                plt.savefig(save_path, dpi=100)
                plt.close(fig)

    #################################################
    def visualize_grid_attention_effectiveness(self, q_matrices: Dict[int, torch.Tensor], k_matrices: Dict[int, torch.Tensor], attention_matrices: Dict[int, torch.Tensor], target_mass: float = 0.9, block_size: int = 128, grid_size: int = 8, output_dir: str = "grid_attention_effectiveness"):
        """
        Visualizes:
        (a) Original high values in the attention matrix
        (b) Block-wise importance
        (c) Grid attention selected blocks
        Formatted specifically for paper insertion.
        """
        self.base_output_dir = output_dir
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        layers = sorted(q_matrices.keys())
        print(f"Starting Grid Attention Effectiveness Visualization for {len(layers)} layers...")

        for layer_idx in layers:
            layer_dir = os.path.join(self.base_output_dir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            Q_layer = q_matrices[layer_idx]
            K_layer = k_matrices[layer_idx]
            Attn_layer = attention_matrices.get(layer_idx)

            if Attn_layer is None:
                continue

            num_heads = Q_layer.shape[1]

            for head_idx in range(num_heads):
                attn_head = Attn_layer[0, head_idx]
                seq_len = attn_head.shape[0]

                # --- (1) Original High Values (Top-P Mass) ---
                sorted_vals, sorted_indices = torch.sort(attn_head, descending=True, dim=-1)
                cumsum_vals = torch.cumsum(sorted_vals, dim=-1)
                cumsum_without_self = torch.cat([torch.zeros_like(cumsum_vals[..., :1]), cumsum_vals[..., :-1]], dim=-1)
                mask_remove = cumsum_without_self >= target_mass
                mask_remove[..., 0] = 0

                keep_mask_original = torch.zeros_like(attn_head, dtype=torch.bool)
                keep_mask_original.scatter_(dim=-1, index=sorted_indices, src=~mask_remove)

                post_softmax_vis = torch.full_like(attn_head, float('nan'))
                post_softmax_vis[keep_mask_original] = 1.0

                # --- (2) Grid Attention Mask ---
                q_input = Q_layer[:, head_idx:head_idx+1, :, :]
                k_input = K_layer[:, head_idx:head_idx+1, :, :]
                block_mask, _ = create_block_mask(
                    q_input, k_input, block_size=block_size, patch_size=grid_size,
                    causal=True, cumulative_p_threshold=target_mass,
                    chunk_size=4096, keep_sink=True, keep_local=True
                )
                block_mask_np = block_mask[0, 0].detach().cpu().numpy().astype(float)

                # ================= Plotting =================
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                plt.subplots_adjust(bottom=0.25, wspace=0.3)

                # Plot (a)
                axes[0].imshow(post_softmax_vis.detach().cpu().numpy(), aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1.0)
                axes[0].set_title("(a) Full Attention High Scores", y=-0.30, fontsize=20)
                axes[0].set_xlabel("Key Position", fontsize=18, fontweight='bold', color='black', labelpad=12)
                axes[0].set_ylabel("Query Position", fontsize=18, fontweight='bold', color='black')

                # Plot (b)
                n_mask_q, n_mask_k = block_mask_np.shape
                extent_c = [0, n_mask_k * block_size, n_mask_q * block_size, 0]
                im_b = axes[1].imshow(block_mask_np, aspect='auto', cmap='Blues', interpolation='nearest', extent=extent_c, vmin=0, vmax=1.0)
                axes[1].set_title("(b) DBA Estimator Result", y=-0.30, fontsize=20)
                axes[1].set_xlabel("Key Position", fontsize=18, fontweight='bold', color='black', labelpad=12)

                # Legend for DBA Estimator
                cbar = fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
                cbar.set_ticks([0.1, 0.9])
                cbar.set_ticklabels(['Ignored\n(Lighter Blue)', 'Selected\n(Darker Blue)'], fontsize=14, fontweight='bold')

                for ax in axes:
                    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
                    ax.set_xlim(0, seq_len)
                    ax.set_ylim(seq_len, 0)
                    ax.set_aspect('equal')
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_color('black')

                # Save the figure tightly
                save_path = os.path.join(layer_dir, f"head_{head_idx:02d}.pdf")
                plt.savefig(save_path, format='pdf', bbox_inches='tight')
                plt.close(fig)
