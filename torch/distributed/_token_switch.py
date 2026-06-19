# mypy: allow-untyped-defs
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


def _import_nccl_ep() -> Any:
    # The EP bindings live in the optional torch._nccl_ep extension, built only
    # with USE_NCCL_EP. It statically links libnccl_ep (its nccl* symbols bind to
    # torch's own NCCL) and bakes in the JIT header paths, so it is fully
    # self-contained -- no libnccl_ep.so to preload and no nccl4py dependency.
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep
    except ImportError as e:
        raise ImportError(
            "torch._nccl_ep is unavailable; this PyTorch was not built with "
            "USE_NCCL_EP."
        ) from e

    return _ep


@dataclass(frozen=True, slots=True)
class Routing:
    handle: object
    topk_idx: torch.Tensor


class _DispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _N, H = tokens.shape
        K = topk_weights.shape[1]
        out_tokens = tokens.new_zeros(max_recv_tokens, H)
        out_topk_weights = topk_weights.new_zeros(max_recv_tokens, K)
        out_topk_idx = routing.topk_idx.new_zeros(max_recv_tokens, K)
        ts._dispatch(
            routing, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx
        )
        ctx.ts = ts
        ctx.routing = routing
        ctx.tokens_shape = tokens.shape
        return out_tokens, out_topk_weights, out_topk_idx

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any,
        grad_out_tokens: torch.Tensor,
        grad_out_topk_weights: torch.Tensor,
        grad_out_topk_idx: torch.Tensor,
    ) -> tuple[None, None, torch.Tensor, None, None]:
        grad_tokens = grad_out_tokens.new_zeros(ctx.tokens_shape)
        ctx.ts._combine(ctx.routing, grad_out_tokens.contiguous(), grad_tokens)
        return None, None, grad_tokens, None, None


class _CombineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        expert_tokens: torch.Tensor,
    ) -> torch.Tensor:
        N = routing.topk_idx.shape[0]
        H = expert_tokens.shape[1]
        out_tokens = expert_tokens.new_zeros(N, H)
        ts._combine(routing, expert_tokens, out_tokens)
        ctx.ts = ts
        ctx.routing = routing
        ctx.expert_shape = expert_tokens.shape
        ctx.expert_dtype = expert_tokens.dtype
        ctx.top_k = routing.topk_idx.shape[1]
        return out_tokens

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_out_tokens: torch.Tensor
    ) -> tuple[None, None, torch.Tensor]:
        M, H = ctx.expert_shape
        N = grad_out_tokens.shape[0]
        K = ctx.top_k
        dtype = ctx.expert_dtype
        # ncclEpDispatch requires the output buffer sized to the group's
        # max_recv_tokens_per_rank, regardless of what shape expert_tokens had
        # in forward (often a slice like out_tokens[:M]). Allocate full-size,
        # run dispatch, then slice to ctx.expert_shape so the returned grad
        # matches the input that produced it.
        max_recv = ctx.ts._max_recv_tokens_per_rank
        grad_expert_full = grad_out_tokens.new_zeros(max_recv, H).to(dtype)
        dummy_weights = grad_out_tokens.new_zeros(N, K, dtype=torch.float32)
        dummy_out_weights = grad_out_tokens.new_zeros(max_recv, K, dtype=torch.float32)
        dummy_out_idx = ctx.routing.topk_idx.new_zeros(max_recv, K)
        ctx.ts._dispatch(
            ctx.routing,
            grad_out_tokens.to(dtype).contiguous(),
            dummy_weights,
            grad_expert_full,
            dummy_out_weights,
            dummy_out_idx,
        )
        return None, None, grad_expert_full[:M].contiguous()


class TokenSwitch(abc.ABC):
    """Abstract token routing switch (e.g. expert-parallel dispatch / combine).

    Typical usage: :meth:`create_routing`, then :meth:`dispatch` / :meth:`combine`.
    """

    @abc.abstractmethod
    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32, length >= local experts:
        output buffer for per-expert receive counts (NCCL EP ``RECV_EXPERT_COUNTER``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
        expert_tokens_hdl: object | None = None,
    ) -> None:
        raise NotImplementedError

    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int | None = None,
        *,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        With ``out=(out_tokens, out_topk_weights, out_topk_idx)``: writes to the provided
        buffers and returns them; no autograd support.
        Without ``out``: allocates output buffers and returns
        ``(out_tokens, out_topk_weights, out_topk_idx)`` with autograd support.
        ``max_recv_tokens`` is required when ``out`` is not provided.
        ``topk_weights`` receives no gradient (routing metadata).
        """
        if out is not None:
            self._dispatch(routing, tokens, topk_weights, *out)
            return out
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out= is not provided")
        return _DispatchAutograd.apply(
            self, routing, tokens, topk_weights, max_recv_tokens
        )  # type: ignore[return-value]

    def combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        expert_tokens_hdl: object | None = None,
    ) -> torch.Tensor:
        """Gather expert outputs back to token order.

        With ``out=out_tokens``: writes to the provided buffer and returns it;
        no autograd support.
        Without ``out``: allocates an output buffer and returns it with autograd support.
        ``expert_tokens_hdl``: optional ``_SymmetricMemory`` handle for
        ``expert_tokens`` from the NCCL symm_mem backend.  When provided,
        NCCL EP uses the associated NCCL window to P2P-read expert outputs
        from peer ranks without staging through an internal RDMA copy buffer
        (zero-copy combine input).  Caller must barrier the handle before
        calling so all ranks' writes are visible via NVLink.
        """
        if out is not None:
            self._combine(routing, expert_tokens, out, expert_tokens_hdl)
            return out
        if expert_tokens_hdl is not None:
            # Windowed path requires a concrete output buffer; autograd is not
            # supported here since this path is intended for inference.
            N = routing.topk_idx.shape[0]
            H = expert_tokens.shape[1]
            out_buf = expert_tokens.new_zeros(N, H)
            self._combine(routing, expert_tokens, out_buf, expert_tokens_hdl)
            return out_buf
        return _CombineAutograd.apply(self, routing, expert_tokens)  # type: ignore[return-value]


class TokenSwitchNCCLExpertMajor(TokenSwitch):
    """TokenSwitch with expert-major dispatch output.

    Tokens dispatched to this rank are placed contiguous-per-expert in the output
    buffer instead of the flat (arrival-order) layout used by :class:`TokenSwitchNCCL`.
    The ``expert_offsets`` tensor returned by :meth:`dispatch_expert_major` is the
    cumulative padded per-expert slot count and can be passed directly as ``offs=``
    to :func:`torch.nn.functional.grouped_mm`.

    After the grouped GEMM, the expert-major output (with topk weights pre-applied)
    goes straight into :meth:`combine` — no re-sorting needed.
    """

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
        alignment: int = 1,
    ) -> None:
        self._ep = _import_nccl_ep()
        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        self._num_local_experts = num_local_experts
        self._alignment = alignment
        self._group = self._ep._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing; ``per_expert_token_counts`` is ignored (expert-major
        uses internally-managed counters via the handle config)."""
        handle = self._ep._NcclEpHandle.create_expert_major(
            self._group,
            topk_idx,
            self._num_local_experts,
            self._alignment,
        )
        return Routing(handle=handle, topk_idx=topk_idx)

    def dispatch_expert_major(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        out: tuple[torch.Tensor, torch.Tensor] | None = None,
        out_tokens_hdl: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts in expert-major order.

        Returns ``(out_tokens, out_topk_weights, expert_offsets)`` where:

        * ``out_tokens`` — shape ``[num_recv_slots, hidden]``, expert-major ordered.
        * ``out_topk_weights`` — shape ``[num_recv_slots]``, one weight per slot.
        * ``expert_offsets`` — shape ``[num_local_experts]`` int32, cumulative padded
          per-expert slot counts.  Pass as ``offs=`` to
          :func:`torch.nn.functional.grouped_mm`.

        With ``out=(out_tokens, out_topk_weights)``: writes into the provided buffers
        (no autograd support).  Without ``out``: allocates output buffers sized to
        ``max_recv_tokens_per_rank``.
        ``out_tokens_hdl``: optional ``_SymmetricMemory`` handle for ``out_tokens``
        from the NCCL symm_mem backend.  When provided, NCCL EP writes dispatch
        output directly into the symmetric buffer without staging through an internal
        RDMA copy buffer (zero-copy dispatch output).
        """
        H = tokens.shape[1]
        if out is None:
            out_tokens = tokens.new_zeros(self._max_recv_tokens_per_rank, H)
            out_topk_weights = torch.zeros(
                self._max_recv_tokens_per_rank,
                dtype=torch.float32,
                device=tokens.device,
            )
        else:
            out_tokens, out_topk_weights = out
        if out_tokens_hdl is not None:
            self._ep._nccl_ep_dispatch_expert_major_windowed(
                routing.handle,
                tokens,
                topk_weights,
                out_tokens,
                out_tokens_hdl,
                out_topk_weights,
            )
        else:
            self._ep._nccl_ep_dispatch_expert_major(
                routing.handle,
                tokens,
                topk_weights,
                out_tokens,
                out_topk_weights,
            )
        expert_offsets = routing.handle.get_expert_offsets()
        return out_tokens, out_topk_weights, expert_offsets

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        # Satisfies the abstract method; expert-major dispatch fills out_tokens
        # and a 1D out_topk_weights slice but leaves out_topk_idx untouched.
        num_recv = out_tokens.shape[0]
        flat_weights = out_topk_weights.new_zeros(num_recv)
        self._ep._nccl_ep_dispatch_expert_major(
            routing.handle, tokens, topk_weights, out_tokens, flat_weights
        )

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
        expert_tokens_hdl: object | None = None,
    ) -> None:
        if expert_tokens_hdl is not None:
            self._ep._nccl_ep_combine_windowed(
                routing.handle, expert_tokens, expert_tokens_hdl, out_tokens
            )
        else:
            self._ep._nccl_ep_combine(routing.handle, expert_tokens, out_tokens)


class TokenSwitchNCCL(TokenSwitch):
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine)."""

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
    ) -> None:
        self._ep = _import_nccl_ep()
        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        self._group = self._ep._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`."""
        handle = self._ep._NcclEpHandle.create(
            self._group,
            topk_idx,
            per_expert_token_counts,
        )
        return Routing(handle=handle, topk_idx=topk_idx)

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        self._ep._nccl_ep_dispatch(
            routing.handle,
            tokens,
            topk_weights,
            out_tokens,
            out_topk_weights,
            out_topk_idx,
        )

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
        expert_tokens_hdl: object | None = None,
    ) -> None:
        self._ep._nccl_ep_combine(
            routing.handle,
            expert_tokens,
            out_tokens,
        )
