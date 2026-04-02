# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel,
    select_unquantized_moe_backend,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import TritonExperts
else:
    TritonExperts = None  # type: ignore


logger = init_logger(__name__)

import os

_CORRUPTION_CHECK = os.environ.get("VLLM_CORRUPTION_CHECK", "0") == "1"
_EXPERT_DEBUG = os.environ.get("VLLM_EXPERT_DEBUG", "0") == "1"
_KERNEL_PROFILE = os.environ.get("VLLM_KERNEL_PROFILE", "0") == "1"

# ── Kernel/copy/wait profiling (VLLM_KERNEL_PROFILE=1) ──
if _KERNEL_PROFILE:
    import time as _time
    _kp_interval = 200  # log every N calls
    _kp_calls = 0
    _kp_kernel_total_us = 0.0
    _kp_wait_total_us = 0.0
    _kp_kernel_count = 0
    _kp_wait_count = 0
    # Overlap window: intra-step dense gap (L_i end → L_{i+1} entry)
    _ow_last_end: "torch.cuda.Event | None" = None  # type: ignore
    _ow_last_layer_idx: int = -1
    _ow_gap_total_us = 0.0
    _ow_gap_count = 0


# --8<-- [start:unquantized_fused_moe]
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend = select_unquantized_moe_backend(
            use_ep=self.moe.moe_parallel_config.use_ep,
            use_dp=self.moe.moe_parallel_config.dp_size > 1,
        )

        # AITER only supports gated activations (silu/gelu), so disable it
        # for non-gated MoE (is_act_and_mul=False)
        self.rocm_aiter_moe_enabled = (
            rocm_aiter_ops.is_fused_moe_enabled() and moe.is_act_and_mul
        )
        self.kernel: mk.FusedMoEModularKernel | None = None
        self._is_monolithic = current_platform.is_cpu() or current_platform.is_xpu()

    @property
    def is_monolithic(self) -> bool:
        return self._is_monolithic

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def allow_inplace(self) -> bool:
        return True

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalize | None:
        if self.unquantized_backend == UnquantizedMoeBackend.AITER:
            return None
        else:
            return super().maybe_make_prepare_finalize(routing_tables)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (
            envs.VLLM_ROCM_MOE_PADDING
            and current_platform.is_rocm()
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0
        ):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()

        return weight

    def _setup_kernel(
        self,
        layer: Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2 = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            layer=layer,
            w13_weight=w13,
            w2_weight=w2,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        # Setup Modular Kernel for TP Case
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        self.kernel, self.use_inplace = make_unquantized_moe_kernel(
            backend=self.unquantized_backend,
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        if self.unquantized_backend == UnquantizedMoeBackend.XPU:
            import intel_extension_for_pytorch as ipex

            ep_rank_start = self.moe.ep_rank * self.moe.num_local_experts
            self.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_weight,
                layer.w2_weight,
                use_prepack=True,
                experts_start_id=ep_rank_start,
            )
        elif self.unquantized_backend == UnquantizedMoeBackend.CPU:
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe

            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.utils import check_cpu_sgl_kernel

                dtype_w13 = layer.w13_weight.dtype
                _, n_w13, k_w13 = layer.w13_weight.size()
                dtype_w2 = layer.w2_weight.dtype
                _, n_w2, k_w2 = layer.w2_weight.size()
                if (
                    envs.VLLM_CPU_SGL_KERNEL
                    and check_cpu_sgl_kernel(n_w13, k_w13, dtype_w13)
                    and check_cpu_sgl_kernel(n_w2, k_w2, dtype_w2)
                ):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(
                        layer.w13_weight
                    )
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(
                        layer.w2_weight
                    )
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    self.cpu_fused_moe: Callable = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    self.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
            else:
                self.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
        elif current_platform.is_cuda_alike():
            self._setup_kernel(
                layer=layer,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
            )

    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def forward_cuda(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.kernel is not None
        # Check if scratch bank is active for THIS layer's call.
        # Gate on both debug flag AND owner-layer match to prevent
        # stale scratch from a different layer leaking through.
        _ec = getattr(layer, '_expert_cache', None)
        _scratch_configured = (
            _ec is not None and _ec._scratch_w13 is not None
        )
        _scratch_active = False
        _active_bank = None
        _active_bank_idx = -1
        _layer_idx = getattr(layer, '_expert_cache_layer_idx', -2)
        # Cutoff: skip bank search for fully-resident layers
        _skip_bank_search = (
            _ec is not None
            and getattr(_ec, '_cutoff_active', False)
            and _layer_idx not in getattr(
                _ec, '_cutoff_tail_layers', set()))
        if (_scratch_configured
                and getattr(_ec, '_scratch_banks', None)
                and not _skip_bank_search):
            from .expert_cache import _BankState
            for _bi, _bk in enumerate(_ec._scratch_banks):
                if (_bk.state == _BankState.READY
                        and _bk.owner_layer_idx == _layer_idx):
                    _active_bank = _bk
                    _active_bank_idx = _bi
                    _bk.state = _BankState.IN_USE  # READY → IN_USE
                    break
            _scratch_active = _active_bank is not None
            # Assert: if cache_map has scratch slots but no bank found,
            # state machine is broken (silent quality corruption).
            # Skip during CUDA graph capture (.item() is illegal).
            if (not _scratch_active and _ec._scratch_threshold > 0
                    and not torch.cuda.is_current_stream_capturing()
                    and not getattr(_ec, '_dormant', True)
                    and not getattr(_ec, '_fixed_tail_active', False)
                    and not getattr(_ec, '_static_topo_active', False)
                    and not getattr(_ec, '_cutoff_active', False)):
                _cm = layer._cache_map
                _has_scratch_slot = (_cm >= _ec._scratch_threshold).any().item()
                if _has_scratch_slot:
                    _states = [(i, b.state.name, b.owner_layer_idx)
                               for i, b in enumerate(_ec._scratch_banks)]
                    raise AssertionError(
                        f"cache_map has scratch slots (>={_ec._scratch_threshold}) "
                        f"but no READY bank for layer {_layer_idx}. "
                        f"Banks: {_states}. "
                        f"This indicates a state machine bug in "
                        f"_consume_prewarm or reserve_scratch.")
        elif _scratch_configured:
            # Legacy single-bank fallback (should not reach after migration)
            _scratch_active = (
                _ec._scratch_in_use
                and _ec._scratch_owner_layer_idx
                    == getattr(layer, '_expert_cache_layer_idx', -2)
            )
        # 보조 진단: dormant regime에서는 active가 아니므로 skip
        if _scratch_active and torch.compiler.is_compiling():
            raise RuntimeError(
                "Scratch MVP: compiled/inductor path not supported yet. "
                "Set VLLM_EXPERT_SCRATCH_CAPACITY=0 or disable torch.compile."
            )
        if _EXPERT_DEBUG and not getattr(self, '_dispatch_logged', False):
            self._dispatch_logged = True
            logger.info(
                "MoE dispatch: method=forward_cuda, backend=%s, "
                "kernel_type=%s, scratch_available=%s",
                self.unquantized_backend,
                type(self.kernel).__name__ if self.kernel else "None",
                _ec is not None and _ec._scratch_w13 is not None,
            )

        # KernelProfile: disable during CUDA graph capture (Events/sync illegal)
        _kp_active = (_KERNEL_PROFILE
                      and not torch.cuda.is_current_stream_capturing())

        # Overlap window: record entry, only count L_i → L_{i+1} (not cross-step)
        if _kp_active:
            _ev_entry = torch.cuda.Event(enable_timing=True)
            _ev_entry.record()
            _cur_layer = getattr(layer, '_expert_cache_layer_idx', -1)

        _w2_ready_event = None
        if _scratch_active and _active_bank is not None:
            # Split prefetch: wait on w13_ready_event only (w2 may still
            # be copying). Pass w2 ready_event to kernel for deferred wait.
            if _kp_active:
                _ev_wait_s = torch.cuda.Event(enable_timing=True)
                _ev_wait_e = torch.cuda.Event(enable_timing=True)
                _ev_wait_s.record()
            _has_split = (getattr(_active_bank, 'w13_ready_event', None)
                          is not None
                          and getattr(_ec, '_tp_size', 99) <= 2
                          and (getattr(_ec, '_cutoff_active', False)
                               or getattr(_ec, '_static_topo_active', False)))
            if _has_split:
                torch.cuda.current_stream().wait_event(
                    _active_bank.w13_ready_event)
                _w2_ready_event = _active_bank.ready_event
            else:
                torch.cuda.current_stream().wait_event(
                    _active_bank.ready_event)
            if _kp_active:
                _ev_wait_e.record()

        # Step-boundary / Cutoff: trigger next-layer prefetch (overlap H2D ∥ kernel)
        # MUST be outside _scratch_active: resident-only layers (no bank)
        # still need to prefetch for the next tail layer.
        if getattr(_ec, '_static_topo_active', False):
            _ec.step_prefetch_next(
                getattr(layer, '_expert_cache_layer_idx', -1))
        elif getattr(_ec, '_cutoff_active', False):
            # Skip prefetch if next layer is fully resident
            if (_layer_idx + 1) in getattr(
                    _ec, '_cutoff_tail_layers', set()):
                _ec.cutoff_prefetch_next(_layer_idx)

        # ── CacheMap -1 detection (VLLM_CORRUPTION_CHECK=1 only) ──
        # GPU sync via .item() — disabled by default for performance.
        if (_CORRUPTION_CHECK
                and _ec is not None and layer._cache_map is not None
                and not torch.cuda.is_current_stream_capturing()
                and not getattr(_ec, '_fixed_tail_active', False)
                and not getattr(_ec, '_static_topo_active', False)
                and not getattr(_ec, '_cutoff_active', False)):
            _cm = layer._cache_map
            _routed = topk_ids.flatten()
            _routed_valid = _routed[_routed >= 0]
            if _routed_valid.numel() > 0:
                _mapped = _cm[_routed_valid]
                _n_neg = (_mapped == -1).sum().item()
                if _n_neg > 0:
                    _lidx = getattr(layer, '_expert_cache_layer_idx', -1)
                    _uniq_gids = _routed_valid[_mapped == -1].unique()
                    if not hasattr(self, '_cm_miss_logged'):
                        self._cm_miss_logged = 0
                    if self._cm_miss_logged < 50:
                        self._cm_miss_logged += 1
                        logger.warning(
                            "[CacheMap-Miss] L%d: %d/%d routed tokens "
                            "hit cache_map=-1 (%d unique gids). "
                            "scratch=%s gids=%s",
                            _lidx, _n_neg, _routed_valid.numel(),
                            _uniq_gids.numel(),
                            "active" if _scratch_active else "OFF",
                            _uniq_gids.tolist()[:10])

        if _kp_active:
            _ev_k_s = torch.cuda.Event(enable_timing=True)
            _ev_k_e = torch.cuda.Event(enable_timing=True)
            _ev_k_s.record()

        result = self.kernel(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=self.use_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,  # _cache_map, in-place patched
            scratch_w13=_active_bank.w13 if _scratch_active else None,
            scratch_w2=_active_bank.w2 if _scratch_active else None,
            scratch_threshold=_ec._scratch_threshold if _scratch_active else 0,
            w2_ready_event=_w2_ready_event,
        )

        if _kp_active:
            _ev_k_e.record()

        # ── NaN/Inf probe (VLLM_CORRUPTION_CHECK=1 only) ──
        # GPU sync via .item() — disabled by default for performance.
        if (_CORRUPTION_CHECK
                and _ec is not None
                and not torch.cuda.is_current_stream_capturing()
                and getattr(_ec, '_batched_d2h_ready', False)
                and (not (getattr(_ec, '_static_topo_active', False)
                         or getattr(_ec, '_cutoff_active', False))
                     or os.environ.get("VLLM_STATIC_PROBE", "0") == "1")):
            _out = result[0] if isinstance(result, tuple) else result
            if _out is not None and _out.numel() > 0:
                _has_nan = torch.isnan(_out).any().item()
                _has_inf = torch.isinf(_out).any().item()
                if _has_nan or _has_inf:
                    _lidx = getattr(layer, '_expert_cache_layer_idx', -1)
                    if not hasattr(self, '_naninf_logged'):
                        self._naninf_logged = 0
                    if self._naninf_logged < 50:
                        self._naninf_logged += 1
                        _nan_cnt = torch.isnan(_out).sum().item()
                        _inf_cnt = torch.isinf(_out).sum().item()
                        logger.warning(
                            "[NaN-Inf] L%d: nan=%d inf=%d shape=%s "
                            "scratch=%s tokens=%d",
                            _lidx, _nan_cnt, _inf_cnt,
                            list(_out.shape),
                            "active" if _scratch_active else "OFF",
                            topk_ids.shape[0])

        if _scratch_active:
            # Restore _cache_map + record done event on compute stream.
            # Must happen here (after kernel, on same stream), not at
            # Python function boundary.
            _ec.release_scratch(layer, _active_bank_idx)

        if _kp_active:
            global _kp_calls, _kp_kernel_total_us, _kp_wait_total_us
            global _kp_kernel_count, _kp_wait_count
            global _ow_last_end, _ow_last_layer_idx
            global _ow_gap_total_us, _ow_gap_count
            _kp_calls += 1
            # Sync only at log interval to minimize overhead
            if _kp_calls % _kp_interval == 0:
                torch.cuda.synchronize()
                _k_us = _ev_k_s.elapsed_time(_ev_k_e) * 1000  # ms→μs
                _kp_kernel_total_us += _k_us
                _kp_kernel_count += 1
                if _scratch_active and _active_bank is not None:
                    _w_us = _ev_wait_s.elapsed_time(_ev_wait_e) * 1000
                    _kp_wait_total_us += _w_us
                    _kp_wait_count += 1
                # Overlap window: only count adjacent layers (L_i → L_{i+1})
                # Excludes L47→L0 cross-step gaps and phase transitions
                if (_ow_last_end is not None
                        and _cur_layer == _ow_last_layer_idx + 1):
                    _g_us = _ow_last_end.elapsed_time(_ev_entry) * 1000
                    _ow_gap_total_us += _g_us
                    _ow_gap_count += 1
                if _kp_calls % (_kp_interval * 48) == 0:
                    # Full model pass worth of samples
                    _avg_k = (_kp_kernel_total_us / _kp_kernel_count
                              if _kp_kernel_count else 0)
                    _avg_w = (_kp_wait_total_us / _kp_wait_count
                              if _kp_wait_count else 0)
                    _avg_g = (_ow_gap_total_us / _ow_gap_count
                              if _ow_gap_count else 0)
                    logger.info(
                        "[KernelProfile] calls=%d "
                        "kernel_avg=%.0fμs (n=%d) "
                        "wait_avg=%.0fμs (n=%d) "
                        "moe_gap_avg=%.0fμs (n=%d, adj L_i→L_{i+1} only)",
                        _kp_calls, _avg_k, _kp_kernel_count,
                        _avg_w, _kp_wait_count,
                        _avg_g, _ow_gap_count)
                    _kp_kernel_total_us = 0.0
                    _kp_wait_total_us = 0.0
                    _kp_kernel_count = 0
                    _kp_wait_count = 0
                    _ow_gap_total_us = 0.0
                    _ow_gap_count = 0
            # Store this kernel's end event + layer_idx for next gap calc
            _ow_last_end = _ev_k_e
            _ow_last_layer_idx = _cur_layer

        return result

    def forward_monolithic_cpu(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.cpu_fused_moe(
            layer,
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            layer.global_num_experts,
            layer.expert_map,
            layer.custom_routing_function,
            layer.scoring_func,
            layer.routed_scaling_factor,
            layer.e_score_correction_bias,
            layer.apply_router_weight_on_input,
            layer.activation,
        )

    def forward_monolithic_xpu(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.ipex_fusion(
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
        )

    if current_platform.is_cpu():
        forward_native: Callable = forward_monolithic_cpu
        apply_monolithic = forward_monolithic_cpu
    elif current_platform.is_xpu():
        forward_native = forward_monolithic_xpu
        apply_monolithic = forward_monolithic_xpu
    else:
        forward_native = forward_cuda
