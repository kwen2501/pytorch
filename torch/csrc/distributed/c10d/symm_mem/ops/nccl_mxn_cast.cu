#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

#include "reshard_3d_tensor.h"

#include <cstring>

// In-place M-to-N cast (resharding) wrapper around `ncclReshard3D` from
// `third_party/nccl-reshard`.  The caller provides a single
// NCCL-symmetric-memory-allocated buffer `buf` that is large enough to hold
// both the source layout (before the call) and the destination layout
// (after the call).  On entry `buf` contains this rank's source shard
// (laid out as `src_local_shape`); on return it contains this rank's
// destination shard (laid out as `dst_local_shape`).
//
// The source and destination are each described by a 2-D mesh of ranks
// (start_rank, dims) and a per-mesh-dim placement:
//   -1    -> replicate along that mesh dimension
//   >= 0  -> shard tensor dim `p` along that mesh dimension
// Only one mesh dimension may be a SHARD at a time (matching the underlying
// `ncclReshard3DMesh` constraints).

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

namespace {

void fill_mesh_info(
    ::ncclReshard3DMesh& mesh,
    at::IntArrayRef dims,
    int64_t start_rank,
    at::IntArrayRef placement) {
  TORCH_CHECK(
      dims.size() == 2,
      "nccl_mxn_cast: mesh_dims must have length 2, got ", dims.size());
  TORCH_CHECK(
      placement.size() == 2,
      "nccl_mxn_cast: placement must have length 2, got ", placement.size());
  mesh.dims[0] = static_cast<int>(dims[0]);
  mesh.dims[1] = static_cast<int>(dims[1]);
  mesh.start_rank = static_cast<int>(start_rank);
  mesh.placement[0] = static_cast<int>(placement[0]);
  mesh.placement[1] = static_cast<int>(placement[1]);
}

// See third_party/nccl-reshard/python/src/reshard_py.cpp:fixFullyReplicated.
// When both mesh dims are REPLICATE, the C++ reshard layer's
// `computeMeshGroupInfo3D` loses track of the replication dimension; the
// pybind wrapper collapses the mesh to [total, 1] with
// placement=[REPLICATE, SHARD(0)] so that rep_count is correct and the
// shard count is 1 (no-op along tensor dims).  Mirror that here.
void fix_fully_replicated(::ncclReshard3DMesh& mesh) {
  if (mesh.placement[0] == NCCLRESHARD3D_REPLICATE &&
      mesh.placement[1] == NCCLRESHARD3D_REPLICATE) {
    int total = mesh.dims[0] * mesh.dims[1];
    mesh.dims[0] = total;
    mesh.dims[1] = 1;
    mesh.placement[0] = NCCLRESHARD3D_REPLICATE;
    mesh.placement[1] = NCCLRESHARD3D_SHARD(0);
  }
}

} // namespace

void nccl_mxn_cast(
    at::Tensor& buf,
    at::IntArrayRef src_local_shape,
    at::IntArrayRef src_mesh_dims,
    int64_t src_mesh_start_rank,
    at::IntArrayRef src_placement,
    at::IntArrayRef dst_local_shape,
    at::IntArrayRef dst_mesh_dims,
    int64_t dst_mesh_start_rank,
    at::IntArrayRef dst_placement,
    const std::string& group_name) {
// We used some device comm requirements that were added in NCCL 2.29.7 below.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
  TORCH_CHECK(buf.is_cuda(), "nccl_mxn_cast: buf must be a CUDA tensor");
  TORCH_CHECK(buf.is_contiguous(), "nccl_mxn_cast: buf must be contiguous");
  const int ndims = static_cast<int>(src_local_shape.size());
  TORCH_CHECK(
      ndims == 2 || ndims == 3,
      "nccl_mxn_cast: src_local_shape must be 2-D or 3-D, got ",
      ndims);
  TORCH_CHECK(
      static_cast<int>(dst_local_shape.size()) == ndims,
      "nccl_mxn_cast: dst_local_shape rank (", dst_local_shape.size(),
      ") must match src_local_shape rank (", ndims, ")");

  // A rank is considered to play the "src" role if its src_local_shape has
  // all positive dims, and similarly for "dst".  Passing an empty/zero shape
  // (numel == 0) on one side signals that this rank is not a source (or not
  // a destination) in this reshard.  The library derives the actual role
  // from the mesh ranges internally; the shape arrays are used only for
  // dim inference on the side(s) where this rank participates.
  int64_t src_numel = 1;
  int64_t dst_numel = 1;
  for (int d = 0; d < ndims; ++d) {
    TORCH_CHECK(
        src_local_shape[d] >= 0 && dst_local_shape[d] >= 0,
        "nccl_mxn_cast: local shapes must be non-negative, got src=",
        src_local_shape, ", dst=", dst_local_shape);
    src_numel *= src_local_shape[d];
    dst_numel *= dst_local_shape[d];
  }
  const bool is_src_role = src_numel > 0;
  const bool is_dst_role = dst_numel > 0;
  TORCH_CHECK(
      is_src_role || is_dst_role,
      "nccl_mxn_cast: at least one of src_local_shape or dst_local_shape "
      "must be non-empty; got src=", src_local_shape,
      ", dst=", dst_local_shape);
  int64_t required_numel = 0;
  if (is_src_role) {
    required_numel = std::max(required_numel, src_numel);
  }
  if (is_dst_role) {
    required_numel = std::max(required_numel, dst_numel);
  }
  TORCH_CHECK(
      buf.numel() >= required_numel,
      "nccl_mxn_cast: buf.numel() (", buf.numel(),
      ") must be >= required ", required_numel,
      " (src_numel=", src_numel, ", dst_numel=", dst_numel, ")");

  // The buffer must live in NCCL symmetric memory so that `ncclReshard3D`
  // can issue remote loads/stores against the registered window.
  auto symm_mem = c10d::symmetric_memory::rendezvous(buf, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_mxn_cast: buf must be allocated via NCCL symmetric memory "
      "(use symm_mem.empty with NCCL backend)");
  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(
      nccl_hdl != nullptr,
      "nccl_mxn_cast: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(buf.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = buf.device();

  auto& manager = c10d::symmetric_memory::NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);

  // Default CTA count matches the library's benchmark default.
  constexpr int kNumCtas = 16;

  // Create or look up a devcomm dedicated to mxn_cast.  Requirements here
  // mirror `reshard_3d_tensor.cu`'s non-staging path: LSA barriers + a GIN
  // signal slot per source rank per CTA.  The cache key includes the src
  // mesh size so that calls with different-sized source meshes get
  // correctly-sized devcomms (a smaller cached devcomm could under-provision
  // ginSignalCount for a later larger cast).
  const int src_total =
      static_cast<int>(src_mesh_dims.at(0) * src_mesh_dims.at(1));
  const int gin_signal_count = src_total * kNumCtas;
  const std::string devcomm_key =
      std::string("nccl_mxn_cast:") + std::to_string(src_total);
  auto devcomm_opt = manager.get_devcomm(group_name, devcomm_key);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kNumCtas;
    reqs.railGinBarrierCount = kNumCtas;
    reqs.ginSignalCount = gin_signal_count;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    reqs.ginContextCount = kNumCtas;
    ncclDevComm devcomm;
    std::memset(&devcomm, 0, sizeof(devcomm));
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_mxn_cast");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, devcomm_key);
  }
  ncclDevComm& devcomm = devcomm_opt->get();

  // Pack mesh specifications in the format expected by the reshard library.
  ::ncclReshard3DMesh src_mesh_info{};
  ::ncclReshard3DMesh dst_mesh_info{};
  fill_mesh_info(
      src_mesh_info, src_mesh_dims, src_mesh_start_rank, src_placement);
  fill_mesh_info(
      dst_mesh_info, dst_mesh_dims, dst_mesh_start_rank, dst_placement);
  fix_fully_replicated(src_mesh_info);
  fix_fully_replicated(dst_mesh_info);

  // The library expects dims where the innermost dim has been multiplied by
  // the element size (matches `ncclReshard3DSimple`'s behavior before it
  // dispatches to `ncclReshard3D`).
  const size_t element_size = static_cast<size_t>(buf.element_size());
  size_t src_dims_bytes[MAX_TENSOR_DIMS] = {0};
  size_t dst_dims_bytes[MAX_TENSOR_DIMS] = {0};
  for (int d = 0; d < ndims; ++d) {
    src_dims_bytes[d] = static_cast<size_t>(src_local_shape[d]);
    dst_dims_bytes[d] = static_cast<size_t>(dst_local_shape[d]);
  }
  src_dims_bytes[ndims - 1] *= element_size;
  dst_dims_bytes[ndims - 1] *= element_size;

  ncclWindow_t window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_mxn_cast: NCCL window is null");

  // In-place: the same buffer serves as both the source and the destination
  // on dual-role ranks.  On ranks that only appear in the src mesh we pass
  // nullptr for the dst buffer (and vice versa) so the library's source-only
  // / dest-only code paths handle shape inference correctly.  Note that the
  // library ultimately derives is_source/is_dest from the mesh ranges, not
  // from these pointers, but passing nullptr is the benchmark-established
  // convention for indicating absence.
  void* data = buf.data_ptr();
  void* src_data = is_src_role ? data : nullptr;
  void* dst_data = is_dst_role ? data : nullptr;

  C10D_NCCL_CHECK(
      ::ncclReshard3D(
          comm,
          window,
          src_data,
          src_dims_bytes,
          ndims,
          &src_mesh_info,
          dst_data,
          dst_dims_bytes,
          &dst_mesh_info,
          &devcomm,
          kNumCtas,
          /*elements_per_chunk=*/0,
          stream,
          /*lsa_window=*/nullptr,
          /*node_local_rank=*/-1,
          /*node_local_size=*/-1),
      "ncclReshard3D failed in nccl_mxn_cast");
#else
  (void)buf;
  (void)src_local_shape;
  (void)src_mesh_dims;
  (void)src_mesh_start_rank;
  (void)src_placement;
  (void)dst_local_shape;
  (void)dst_mesh_dims;
  (void)dst_mesh_start_rank;
  (void)dst_placement;
  (void)group_name;
  TORCH_CHECK(
      false,
      "nccl_mxn_cast requires NCCL >= 2.29.7 for device comm requirements");
#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
}

} // namespace c10d::nccl_extension
