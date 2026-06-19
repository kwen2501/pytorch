#!/usr/bin/env python3
"""
Demo: NCCL-EP dispatch/combine with symmetric memory buffers.

Run with:
    torchrun --nproc-per-node=<N_GPUS> demo_ep_symm_mem.py

Both the dispatch output (expert inputs) and the combine input (expert
outputs / scaled by routing weights) live in symmetric memory, so NCCL EP
can P2P-read them from peer ranks via NVLink without staging through an
internal RDMA copy buffer (zero-copy windowed dispatch output and combine
input).

The NCCL symm_mem backend (set_backend("NCCL")) must be active before any
symm_mem use so that allocations are backed by ncclMemAlloc and carry an
ncclWindow_t.  The window is passed to nccl_ep_dispatch_expert_major_windowed
and nccl_ep_combine_windowed, which set desc.win_hdl instead of desc.data.

The routing-weight multiply is done out-of-place inside a use_mem_pool
context, landing the result (scaled) in symm_mem.  grouped_mm output stays
in regular CUDA memory -- correct for training since its backward needs the
inputs, and the out-of-place multiply is autograd-safe (no in-place mutation).

NCCLSymmetricMemory.barrier() is NYI; the NCCL collectives (dispatch,
combine) provide the necessary memory ordering.

dispatch_in (tokens) and combine_out (combined) are regular tensors -- NCCL EP
reads/writes them only locally and they need no P2P visibility.

Pipeline per forward step:
    tokens [N, H]                            (regular CUDA, dispatch input)
        |
    NCCL-EP dispatch_expert_major
        |
    dispatch_out [max_recv, H]               (symm_mem: expert inputs, P2P-readable)
    dispatch_out_weights [max_recv]
    expert_offsets [num_local_E]             (grouped_mm offs=)
        |
        |  (dispatch collective provides ordering; NCCLSymmetricMemory.barrier NYI)
    gemm_out = grouped_mm(dispatch_out, weights, offs=expert_offsets)
                                             (regular CUDA; kept for backward)
        |
    with use_mem_pool(symm_pool):
        scaled = gemm_out * topk_weights     (new tensor -> symm_mem; autograd-safe)
        |
    combine_symm_hdl = rendezvous(scaled)    (O(1) cache hit after step 1)
        |                                    (NCCLSymmetricMemory.barrier NYI;
        |                                     combine collective provides ordering)
    NCCL-EP combine(scaled, hdl=combine_symm_hdl)  <- windowed: desc.win_hdl set,
        |                                              no RDMA staging copy
    combined [N, H]                          (regular CUDA, combine output)
"""

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem
from torch.distributed._token_switch import TokenSwitchNCCLExpertMajor
from torch.nn.functional import grouped_mm


# ---------- configuration ----------
NUM_TOKENS = 512  # tokens per rank per step
HIDDEN = 256  # token hidden dim
TOP_K = 2  # top-k routing per token
ALIGNMENT = 16  # per-expert zone alignment for grouped_mm
DTYPE = torch.bfloat16


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Force NCCL comm initialization; ProcessGroupNCCL creates it lazily and
    # ncclEpCreateGroup needs a valid ncclComm_t.
    dist.barrier()

    # Must be set before any symm_mem use (set_backend cannot change after first
    # allocation).  NCCL-backend blocks are backed by ncclMemAlloc and carry an
    # ncclWindow_t, enabling nccl_ep_combine_windowed.
    # NCCLSymmetricMemory.barrier() is NYI; we rely on the NCCL collectives
    # (dispatch, combine) to provide the necessary memory ordering before
    # diagnostic get_buffer reads.
    symm_mem.set_backend("NCCL")

    num_experts = 4 * world_size
    num_local_experts = num_experts // world_size  # = 4

    max_recv_tokens = TOP_K * NUM_TOKENS * world_size
    # +1: expert_offsets[-1] < mat_a.size(0) required by grouped_mm.
    dispatch_buf_rows = max_recv_tokens + ALIGNMENT * num_local_experts + 1

    ts = TokenSwitchNCCLExpertMajor(
        process_group=dist.group.WORLD,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        max_dispatch_tokens_per_rank=NUM_TOKENS,
        max_recv_tokens_per_rank=dispatch_buf_rows,
        max_token_bytes=HIDDEN * torch.finfo(DTYPE).bits // 8,
        alignment=ALIGNMENT,
    )

    # Dispatch output: symmetric memory, pre-allocated and rendezvous'd once.
    dispatch_out = symm_mem.empty(dispatch_buf_rows, HIDDEN, dtype=DTYPE, device=device)
    dispatch_symm_hdl = symm_mem.rendezvous(dispatch_out, group=dist.group.WORLD)

    dispatch_out_weights = torch.zeros(
        dispatch_buf_rows, dtype=torch.float32, device=device
    )

    expert_weights = torch.randn(
        num_local_experts, HIDDEN, HIDDEN, dtype=DTYPE, device=device
    )

    symm_pool = symm_mem.get_mem_pool(device)

    for step in range(3):
        tokens = torch.randn(NUM_TOKENS, HIDDEN, dtype=DTYPE, device=device)
        topk_idx = torch.randint(
            0, num_experts, (NUM_TOKENS, TOP_K), dtype=torch.int64, device=device
        )
        topk_weights = torch.rand(
            NUM_TOKENS, TOP_K, dtype=torch.float32, device=device
        ).softmax(dim=-1)

        routing = ts.create_routing(topk_idx)

        # 1. Dispatch into symmetric memory via NCCL window (zero-copy output).
        dispatch_out.zero_()
        dispatch_out_weights.zero_()
        _, _, expert_offsets = ts.dispatch_expert_major(
            routing,
            tokens,
            topk_weights,
            out=(dispatch_out, dispatch_out_weights),
            out_tokens_hdl=dispatch_symm_hdl,
        )

        # 2. Grouped GEMM into regular CUDA memory (kept for backward).
        gemm_out = grouped_mm(dispatch_out, expert_weights, offs=expert_offsets)

        # 3. Scale by routing weights into symm_mem via the pool context.
        #    Out-of-place multiply is autograd-safe (no in-place mutation of a
        #    tensor grouped_mm's backward may need).  The pool reuses two blocks
        #    in alternation; rendezvous is O(1) after the first two steps.
        with torch.cuda.use_mem_pool(symm_pool):
            scaled = gemm_out * dispatch_out_weights.unsqueeze(-1)

        # 4. Rendezvous scaled for the windowed combine.  No explicit barrier:
        #    NCCLSymmetricMemory.barrier is NYI; the combine collective provides
        #    the necessary ordering.
        combine_symm_hdl = symm_mem.rendezvous(scaled, group=dist.group.WORLD)

        # 5. Windowed combine: NCCL EP uses the NCCL window from combine_symm_hdl
        #    to P2P-read scaled from peer ranks without an RDMA staging copy.
        combined = ts.combine(routing, scaled, expert_tokens_hdl=combine_symm_hdl)

        if rank == 0:
            num_recv = routing.handle.get_num_recv_tokens()
            print(
                f"step {step}: recv {num_recv} slots "
                f"(expert_offsets {expert_offsets.cpu().tolist()}) "
                f"-> combined {list(combined.shape)}"
            )

    # Destroy EP group before the NCCL comm; ncclEpGroupDestroy needs a live comm.
    del ts
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
