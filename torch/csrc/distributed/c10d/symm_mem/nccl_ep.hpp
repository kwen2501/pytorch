#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <optional>

namespace c10d::symmetric_memory {
class SymmetricMemory;
}

namespace c10d::nccl_ep {

struct NcclEpGroup : c10::intrusive_ptr_target {
  void* group{nullptr}; // ncclEpGroup_t, opaque to avoid including nccl_ep.h

  NcclEpGroup() = default;
  ~NcclEpGroup();
};

struct NcclEpHandle : c10::intrusive_ptr_target {
  void* handle{nullptr}; // ncclEpHandle_t, opaque
  // The library stashes topk_idx's device pointer on the handle (per nccl_ep.h:
  // "User-owned (do not free). LL reads directly; HT uses cached
  // hybridep.topk_idx"). recv_total_counter is allocated by us and read back
  // by nccl_ep_handle_get_num_recv_tokens. Keep both alive for the handle's
  // lifetime so nccl_ep can't read freed memory.
  at::Tensor topk_idx;
  at::Tensor recv_total_counter;
  // Expert-major layout only: per-expert recv counts and cumulative offsets.
  // Written by ncclEpCreateHandle; expert_offsets maps directly to grouped_mm's
  // offs=.
  at::Tensor expert_counters; // [num_local_experts] int32
  at::Tensor
      expert_offsets; // [num_local_experts] int32, prefix sums of padded counts

  NcclEpHandle(void* handle, at::Tensor topk_idx, at::Tensor recv_total_counter)
      : handle(handle),
        topk_idx(std::move(topk_idx)),
        recv_total_counter(std::move(recv_total_counter)) {}
  NcclEpHandle(
      void* handle,
      at::Tensor topk_idx,
      at::Tensor recv_total_counter,
      at::Tensor expert_counters,
      at::Tensor expert_offsets)
      : handle(handle),
        topk_idx(std::move(topk_idx)),
        recv_total_counter(std::move(recv_total_counter)),
        expert_counters(std::move(expert_counters)),
        expert_offsets(std::move(expert_offsets)) {}
  ~NcclEpHandle();
};

TORCH_API c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>& pg,
    int64_t num_experts,
    int64_t max_dispatch_tokens_per_rank,
    int64_t max_recv_tokens_per_rank,
    int64_t max_token_bytes);

TORCH_API c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>& group,
    const at::Tensor& topk_idx,
    const std::optional<at::Tensor>& recv_expert_counter);

// Expert-major variant: creates a handle with NCCL_EP_LAYOUT_EXPERT_MAJOR.
// Tokens are grouped by local expert in the dispatch output; expert_offsets
// (cumulative padded counts) can be passed directly as grouped_mm's offs=.
// alignment: per-expert zone alignment in tokens (power-of-2; 1 = no padding).
TORCH_API c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle_expert_major(
    const c10::intrusive_ptr<NcclEpGroup>& group,
    const at::Tensor& topk_idx,
    int64_t num_local_experts,
    int64_t alignment);

TORCH_API int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>& handle);

TORCH_API at::Tensor nccl_ep_handle_get_expert_offsets(
    const c10::intrusive_ptr<NcclEpHandle>& handle);

TORCH_API void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    at::Tensor& out_topk_weights,
    at::Tensor& out_topk_idx);

// Expert-major dispatch: out_topk_weights is 1D [num_recv_slots]; no
// out_topk_idx.
TORCH_API void nccl_ep_dispatch_expert_major(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    at::Tensor& out_topk_weights);

// Zero-copy variant: uses the NCCL window from out_tokens_hdl so NCCL EP
// writes dispatch output directly into the symmetric buffer without staging
// through an internal RDMA copy buffer.  Requires the NCCL symm_mem backend
// (set_backend("NCCL") before first symm_mem use).
TORCH_API void nccl_ep_dispatch_expert_major_windowed(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    const c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory>&
        out_tokens_hdl,
    at::Tensor& out_topk_weights);

TORCH_API void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& expert_tokens,
    at::Tensor& out_tokens);

// Zero-copy variant: uses the NCCL window from expert_tokens_hdl so NCCL EP
// can P2P-read expert_tokens from peer ranks without staging through an
// internal RDMA copy buffer.  Requires the NCCL symm_mem backend
// (set_backend("NCCL") before first MemPool use).  Caller must barrier the
// symm_mem handle before calling so all ranks' expert_tokens writes are
// visible via NVLink before NCCL EP reads them.
TORCH_API void nccl_ep_combine_windowed(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& expert_tokens,
    const c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory>&
        expert_tokens_hdl,
    at::Tensor& out_tokens);

} // namespace c10d::nccl_ep
