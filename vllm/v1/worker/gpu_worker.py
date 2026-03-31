# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc
import os
from contextlib import AbstractContextManager, nullcontext
from types import NoneType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

from .utils import request_memory

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class Worker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch/CUDA profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU", "CUDA"],
            )
        elif profiler_config.profiler == "cuda":
            self.profiler = CudaProfilerWrapper(profiler_config)
        else:
            self.profiler = None

        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        self._vmm_pool = None  # VMMPagePool, set by _init_expert_offloading

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        # If the KV cache has just been woken up,
        # the internal state of cache_engine must be reset,
        # especially the FP8 scaling factor.
        if (
            (tags is None or "kv_cache" in tags)
            and self.cache_config.cache_dtype.startswith("fp8")
            and hasattr(self.model_runner, "init_fp8_kv_scales")
        ):
            self.model_runner.init_fp8_kv_scales()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        if self.device_config.device_type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                assert self.local_rank < torch.cuda.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )
                visible_device_count = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                )
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,
            )

            # HACK(woosuk): This is a temporary fix to avoid type errors.
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            from vllm.v1.worker.gpu_model_runner import (
                GPUModelRunner as GPUModelRunnerV1,
            )

            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(
            tag="weights"
        ) and set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)
        # Expert offloading init (after model loading)
        self._init_expert_offloading()

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        self.model_runner.reload_weights()

    def _init_expert_offloading(self):
        """Initialize expert weight offloading. Called after load_model().

        Ported from gpusim: creates ExpertCacheManager, optionally VMMPagePool,
        wires expert cache into FusedMoE layers, and stores references on
        model_runner for use by elastic KV and pre_step().
        """
        import gc
        import math

        from vllm.model_executor.layers.fused_moe.expert_cache import (
            ExpertCacheManager, ExpertOffloadConfig,
        )
        from vllm.model_executor.layers.fused_moe.expert_predictor import (
            ExpertPredictor,
        )
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.utils import replace_parameter

        config = getattr(self.vllm_config, 'expert_offload_config', None)
        vmm_enabled = (
            os.environ.get("VLLM_VMM_EXPERT_POOL", "0") == "1"
        )
        if config is None:
            if (os.environ.get("VLLM_EXPERT_OFFLOAD_ENABLE", "0") != "1"
                    and not vmm_enabled):
                return
            config = ExpertOffloadConfig(
                enable=True,
                max_resident_per_layer=int(
                    os.environ.get("VLLM_EXPERT_MAX_RESIDENT", "512")
                ),
            )
        if not config.enable:
            return

        raw_model = self.model_runner.get_model()
        moe_layers = [
            m for _, m in raw_model.named_modules()
            if isinstance(m, FusedMoE)
        ]
        if not moe_layers:
            return

        ref = moe_layers[0]
        local_E = ref.local_num_experts
        global_E = ref.global_num_experts
        w13_per_expert = ref.w13_weight.shape[1:]
        w2_per_expert = ref.w2_weight.shape[1:]
        dtype = ref.w13_weight.dtype

        max_res = config.max_resident_per_layer
        if max_res <= 0:
            logger.warning(
                "max_resident_per_layer=%d invalid, clamping to 1", max_res)
            max_res = 1

        if max_res >= local_E and not vmm_enabled:
            logger.info(
                "max_resident_per_layer=%d >= local_experts=%d, "
                "no offloading needed", max_res, local_E)
            return

        config.max_resident_per_layer = max_res

        cache = ExpertCacheManager(
            config=config,
            num_layers=len(moe_layers),
            local_num_experts=local_E,
            global_num_experts=global_E,
            expert_w13_shape=tuple(w13_per_expert),
            expert_w2_shape=tuple(w2_per_expert),
            dtype=dtype,
            device=self.device,
        )
        predictor = ExpertPredictor(
            num_layers=len(moe_layers),
            num_local_experts=local_E,
            top_k=ref.top_k,
        )
        max_num_tokens_for_buffers = getattr(
            self.model_runner, 'max_num_tokens', 4096)

        used_offload_aware = False
        for layer_idx, module in enumerate(moe_layers):
            if hasattr(module, '_w13_cpu_store'):
                used_offload_aware = True
                for lid in range(local_E):
                    is_shared = (
                        hasattr(module, 'num_fused_shared_experts')
                        and lid < module.num_fused_shared_experts
                    )
                    cache.register_expert_cpu(
                        layer_idx, lid,
                        module._w13_cpu_store[lid],
                        module._w2_cpu_store[lid],
                        is_shared=is_shared,
                    )
                delattr(module, '_w13_cpu_store')
                delattr(module, '_w2_cpu_store')
                if hasattr(module, '_offload_num_experts'):
                    delattr(module, '_offload_num_experts')
            else:
                for lid in range(local_E):
                    is_shared = (
                        hasattr(module, 'num_fused_shared_experts')
                        and lid < module.num_fused_shared_experts
                    )
                    cache.register_expert_cpu(
                        layer_idx, lid,
                        module.w13_weight[lid], module.w2_weight[lid],
                        is_shared=is_shared,
                    )

                if not vmm_enabled:
                    placeholder_w13 = torch.nn.Parameter(
                        torch.empty(0, dtype=dtype, device='cpu'),
                        requires_grad=False,
                    )
                    placeholder_w2 = torch.nn.Parameter(
                        torch.empty(0, dtype=dtype, device='cpu'),
                        requires_grad=False,
                    )
                    replace_parameter(module, "w13_weight", placeholder_w13)
                    replace_parameter(module, "w2_weight", placeholder_w2)

                    gc.collect()
                    torch.cuda.empty_cache()

                    new_w13 = torch.nn.Parameter(
                        torch.empty(
                            (max_res, *w13_per_expert), dtype=dtype,
                            device=self.device
                        ),
                        requires_grad=False,
                    )
                    new_w2 = torch.nn.Parameter(
                        torch.empty(
                            (max_res, *w2_per_expert), dtype=dtype,
                            device=self.device
                        ),
                        requires_grad=False,
                    )
                    replace_parameter(module, "w13_weight", new_w13)
                    replace_parameter(module, "w2_weight", new_w2)

            cache.register_layer(
                layer_idx, module.w13_weight.data, module.w2_weight.data
            )
            module.set_expert_cache(
                cache, layer_idx,
                max_num_tokens=max_num_tokens_for_buffers)

            if layer_idx % 8 == 0:
                logger.info(
                    "Offload progress: %d/%d layers, GPU alloc: %.1fGB",
                    layer_idx + 1, len(moe_layers),
                    torch.cuda.memory_allocated(self.device) / (1 << 30),
                )

        gc.collect()
        torch.cuda.empty_cache()

        # ── VMM pool init (if enabled) ──
        if vmm_enabled:
            try:
                from vllm.vmm_pool import VMMPagePool, is_vmm_available
                if not is_vmm_available():
                    logger.warning(
                        "VLLM_VMM_EXPERT_POOL=1 but cuda-python VMM "
                        "bindings not available. Falling back to standard.")
                    vmm_enabled = False
            except ImportError:
                logger.warning(
                    "VLLM_VMM_EXPERT_POOL=1 but vllm.vmm_pool not found. "
                    "Falling back to standard.")
                vmm_enabled = False

        if vmm_enabled:
            from vllm.vmm_pool import VMMPagePool

            elem_size = torch.tensor([], dtype=dtype).element_size()
            w13_numel = 1
            for d in w13_per_expert:
                w13_numel *= d
            w2_numel = 1
            for d in w2_per_expert:
                w2_numel *= d
            expert_slot_bytes = (w13_numel + w2_numel) * elem_size

            total_expert_bytes = (
                local_E * len(moe_layers) * expert_slot_bytes
            )
            gpu_total_bytes = torch.cuda.get_device_properties(
                self.device).total_memory
            kv_va_bytes = gpu_total_bytes
            num_layers = len(moe_layers)

            pool = VMMPagePool(
                device_id=self.device.index,
                expert_total_bytes=total_expert_bytes,
                kv_max_bytes=kv_va_bytes,
                expert_slot_bytes=expert_slot_bytes,
                num_layers=num_layers,
                max_slots_per_layer=local_E,
                dtype=dtype,
            )
            self._vmm_pool = pool
            cache.set_vmm_pool(pool)

            w13_bytes = w13_numel * elem_size
            w2_bytes = w2_numel * elem_size
            pool.set_tensor_layout(w13_bytes, w2_bytes)

            phase_c = os.environ.get("VLLM_VMM_PHASE_C", "0") == "1"
            if phase_c:
                cache._phase_c_enabled = True
                pool._phase_c_mode = True
                layout = "slot_aligned"
            else:
                layout = "tight_pack"

            if layout == "slot_aligned":
                num_groups = math.ceil(max_res / pool.group_size)
                expected_pages_per_layer = num_groups * pool.group_pages
            else:
                expected_pages_per_layer = math.ceil(
                    max_res * expert_slot_bytes / pool.page_size)

            logger.info(
                "VMM boot: layout=%s, phase_c=%s, max_res=%d, "
                "group_size=%d, group_pages=%d, pages/layer=%d, "
                "expert_bytes=%d, page_size=%d",
                layout, phase_c, max_res,
                pool.group_size, pool.group_pages,
                expected_pages_per_layer,
                expert_slot_bytes, pool.page_size)

            for layer_idx, module in enumerate(moe_layers):
                old_bytes = (
                    module.w13_weight.data.nbytes
                    + module.w2_weight.data.nbytes
                )
                replace_parameter(module, "w13_weight",
                                  torch.nn.Parameter(
                                      torch.empty(0, dtype=dtype, device='cpu'),
                                      requires_grad=False))
                replace_parameter(module, "w2_weight",
                                  torch.nn.Parameter(
                                      torch.empty(0, dtype=dtype, device='cpu'),
                                      requires_grad=False))
                gc.collect()
                torch.cuda.empty_cache()

                if phase_c:
                    n_pages = pool.allocate_and_map_layer_slot_aligned(
                        layer_idx, max_res)
                else:
                    n_pages = pool.allocate_and_map_layer_tight_pack(
                        layer_idx, max_res)

                if n_pages != expected_pages_per_layer:
                    raise RuntimeError(
                        f"VMM layout mismatch: layer {layer_idx} got "
                        f"{n_pages} pages, expected "
                        f"{expected_pages_per_layer} (layout={layout})")

                w13_stacked, w2_stacked = pool.get_expert_layer_tensors(
                    layer=layer_idx,
                    num_slots=max_res,
                    w13_per_expert=tuple(w13_per_expert),
                    w2_per_expert=tuple(w2_per_expert),
                    dtype=dtype,
                )
                replace_parameter(module, "w13_weight",
                                  torch.nn.Parameter(w13_stacked,
                                                     requires_grad=False))
                replace_parameter(module, "w2_weight",
                                  torch.nn.Parameter(w2_stacked,
                                                     requires_grad=False))
                cache.register_layer(
                    layer_idx, w13_stacked.data, w2_stacked.data)

                if layer_idx % 8 == 0 or layer_idx == num_layers - 1:
                    committed = pool.total_pages
                    logger.info(
                        "VMM layer %d/%d: layout=%s, mapped %d pages "
                        "(committed=%d), GPU=%.1fGB",
                        layer_idx + 1, num_layers, layout,
                        n_pages, committed,
                        torch.cuda.memory_allocated(self.device) / (1 << 30),
                    )

            total_vmm_pages = num_layers * expected_pages_per_layer
            total_vmm_gb = total_vmm_pages * pool.page_size / (1 << 30)
            logger.info(
                "VMM complete: layout=%s, max_res=%d/%d, %d layers, "
                "%d pages (%.1f GiB), pool=%s",
                layout, max_res, local_E, num_layers,
                total_vmm_pages, total_vmm_gb, pool.get_stats())

            self.model_runner._vmm_pool = pool

            cache.populate_initial_cache()
        else:
            cache.populate_initial_cache()

        # Wire cache_map for all layers
        for layer_idx, module in enumerate(moe_layers):
            if hasattr(module, '_cache_map') and module._cache_map is not None:
                cache._update_cache_map(layer_idx, module)

        # Always store MoE layer refs — needed for O2 emap cache,
        # lookahead prefetch, and any future per-layer metadata.
        # Must be unconditional (not gated on scratch/eager config).
        cache.set_moe_layers(moe_layers)
        cache._tp_size = self.vllm_config.parallel_config.tensor_parallel_size

        # ── Scratch bank: shared overflow buffer for hard guarantee ──
        scratch_capacity = int(os.environ.get(
            "VLLM_EXPERT_SCRATCH_CAPACITY", "0"))
        if scratch_capacity > 0:
            # Assert all MoE layers have identical w13/w2 shapes and dtype.
            # Single shared scratch tensor requires this.
            for li, mod in enumerate(moe_layers):
                assert mod.w13_weight.shape[1:] == tuple(w13_per_expert), (
                    f"Layer {li} w13 shape {mod.w13_weight.shape[1:]} != "
                    f"ref {tuple(w13_per_expert)}")
                assert mod.w2_weight.shape[1:] == tuple(w2_per_expert), (
                    f"Layer {li} w2 shape {mod.w2_weight.shape[1:]} != "
                    f"ref {tuple(w2_per_expert)}")
                assert mod.w13_weight.dtype == dtype, (
                    f"Layer {li} dtype {mod.w13_weight.dtype} != {dtype}")

            num_banks = int(os.environ.get("VLLM_SCRATCH_BANKS", "2"))
            if os.environ.get("VLLM_FIXED_TAIL", "0") == "1":
                num_banks = 2  # fixed-tail requires double-buffer
            if os.environ.get("VLLM_STEP_BOUNDARY", "0") == "1":
                num_banks = 2  # step-boundary requires double-buffer
                # PoC scope: TP4 full-cover only
                # (max_tail ≤ scratch_capacity enforced at activation)
                if scratch_capacity < local_E:
                    logger.warning(
                        "[StepBoundary] scratch_capacity=%d < "
                        "local_num_experts=%d. TP4 full-cover requires "
                        "scratch ≥ max possible tail. Non-TP4 setups may "
                        "hit assert at activation.",
                        scratch_capacity, local_E)
            if os.environ.get("VLLM_CUTOFF_BOUNDARY", "0") == "1":
                num_banks = 2  # cutoff-boundary requires double-buffer
            if (os.environ.get("VLLM_FIXED_TAIL", "0") == "1"
                    and os.environ.get("VLLM_STEP_BOUNDARY", "0") == "1"):
                raise ValueError(
                    "VLLM_FIXED_TAIL and VLLM_STEP_BOUNDARY are "
                    "mutually exclusive")
            if (os.environ.get("VLLM_CUTOFF_BOUNDARY", "0") == "1"
                    and os.environ.get("VLLM_FIXED_TAIL", "0") == "1"):
                raise ValueError(
                    "VLLM_FIXED_TAIL and VLLM_CUTOFF_BOUNDARY are "
                    "mutually exclusive")
            if (os.environ.get("VLLM_CUTOFF_BOUNDARY", "0") == "1"
                    and os.environ.get("VLLM_STEP_BOUNDARY", "0") == "1"):
                raise ValueError(
                    "VLLM_STEP_BOUNDARY and VLLM_CUTOFF_BOUNDARY are "
                    "mutually exclusive")
            scratch_w13 = torch.empty(
                (num_banks * scratch_capacity, *w13_per_expert),
                dtype=dtype, device=self.device)
            scratch_w2 = torch.empty(
                (num_banks * scratch_capacity, *w2_per_expert),
                dtype=dtype, device=self.device)
            cache.set_scratch(
                scratch_w13, scratch_w2,
                threshold=max_res,
                capacity=scratch_capacity,
                num_banks=num_banks)
            scratch_bytes = (scratch_w13.nbytes + scratch_w2.nbytes)
            logger.info(
                "Scratch bank: capacity=%d threshold=%d "
                "w13=%s w2=%s %.1fMB GPU=%.1fGB",
                scratch_capacity, max_res,
                list(scratch_w13.shape), list(scratch_w2.shape),
                scratch_bytes / (1 << 20),
                torch.cuda.memory_allocated(self.device) / (1 << 30))

        # Store references for pre_step() and elastic KV
        self.model_runner._expert_cache = cache
        self.model_runner._expert_cache_layers = moe_layers
        if self.model_runner._expert_miss_log:
            cache._collect_routing_freq = True

        # Eager routing: per-layer graph boundary for fresh routing sync
        # Separate prefill/decode control:
        #   VLLM_EXPERT_EAGER_ROUTING_PREFILL="all" or "0,1,2"
        #   VLLM_EXPERT_EAGER_ROUTING_DECODE="0" or "0,47"
        # Legacy single var still supported as both-phase shorthand:
        #   VLLM_EXPERT_EAGER_ROUTING_LAYERS="all" → prefill all + decode all
        def _parse_layer_set(env_val: str) -> set:
            if not env_val:
                return set()
            if env_val.strip().lower() == "all":
                return set(range(len(moe_layers)))
            return {int(x) for x in env_val.split(",") if x.strip()}

        legacy_str = os.environ.get(
            "VLLM_EXPERT_EAGER_ROUTING_LAYERS", "")
        prefill_str = os.environ.get(
            "VLLM_EXPERT_EAGER_ROUTING_PREFILL", "")
        decode_str = os.environ.get(
            "VLLM_EXPERT_EAGER_ROUTING_DECODE", "")

        # Legacy fallback: if new vars not set, use legacy for both phases
        if not prefill_str and not decode_str and legacy_str:
            prefill_str = legacy_str
            decode_str = legacy_str

        prefill_set = _parse_layer_set(prefill_str)
        decode_set = _parse_layer_set(decode_str)

        # Fixed-tail requires ALL layers to have eager decode boundary
        # so that _fixed_tail_prefetch_next() is called for every layer.
        if os.environ.get("VLLM_FIXED_TAIL", "0") == "1":
            all_layers = set(range(len(moe_layers)))
            prefill_set = prefill_set or all_layers
            decode_set = all_layers
            logger.info(
                "VLLM_FIXED_TAIL=1: forcing decode eager routing to ALL "
                "(%d layers)", len(moe_layers))

        if prefill_set or decode_set:
            for layer_idx, mod in enumerate(moe_layers):
                if layer_idx in prefill_set:
                    mod._use_eager_routing_prefill = True
                if layer_idx in decode_set:
                    mod._use_eager_routing_decode = True
            logger.info(
                "Expert eager routing: prefill=%s decode=%s (%d layers)",
                "all" if len(prefill_set) == len(moe_layers)
                else sorted(prefill_set) if prefill_set else "off",
                "all" if len(decode_set) == len(moe_layers)
                else sorted(decode_set) if decode_set else "off",
                len(moe_layers))

        # 3-D: Group boundary — async prefetch between MoE layer groups
        group_size = int(os.environ.get("VLLM_EXPERT_GROUP_SIZE", "0"))
        if group_size > 0:
            num_moe = len(moe_layers)
            cache._group_ranges = [
                list(range(g * group_size,
                           min((g + 1) * group_size, num_moe)))
                for g in range((num_moe + group_size - 1) // group_size)
            ]
            logger.info(
                "Expert group boundaries: size=%d, %d groups, ranges=%s",
                group_size, len(cache._group_ranges),
                [(r[0], r[-1]) for r in cache._group_ranges])

        # Memory accounting
        if vmm_enabled and hasattr(self.model_runner, 'model_memory_usage'):
            old_usage = self.model_runner.model_memory_usage
            self.model_runner.model_memory_usage = (
                torch.cuda.memory_allocated(self.device)
            )
            logger.info(
                "VMM: model_memory_usage %.2fGB -> %.2fGB",
                old_usage / (1 << 30),
                self.model_runner.model_memory_usage / (1 << 30),
            )
        elif not used_offload_aware and hasattr(
                self.model_runner, 'model_memory_usage'):
            freed_bytes = (
                (local_E - max_res) * len(moe_layers)
                * cache.expert_size_bytes
            )
            old_usage = self.model_runner.model_memory_usage
            self.model_runner.model_memory_usage = max(
                0, old_usage - freed_bytes
            )
            logger.info(
                "model_memory_usage adjusted: %.2fGB -> %.2fGB",
                old_usage / (1 << 30),
                self.model_runner.model_memory_usage / (1 << 30),
            )

        freed_bytes = (
            (local_E - max_res) * len(moe_layers) * cache.expert_size_bytes
        )
        logger.info(
            "Expert offloading: %d->%d experts/layer, "
            "%d layers, ~%.1fGB freed",
            local_E, max_res, len(moe_layers), freed_bytes / (1 << 30),
        )

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} "
                f"GiB, reserved {format_gib(kv_cache_memory_bytes)} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return kv_cache_memory_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",
            format_gib(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            format_gib(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %s GiB (total), %s GiB (within requested)",
            format_gib(free_gpu_memory),
            format_gib(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
            scope="local",
        )

        return int(self.available_kv_cache_memory_bytes)

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to GPU memory.

        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %d", max_model_len)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            with allocator.use_memory_pool(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        warmup_sizes = []

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """Get encoder timing stats from model runner."""
        return self.model_runner.get_encoder_timing_stats()

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        iteration_details = compute_iteration_details(scheduler_output)

        annotation = "".join(
            [
                "execute_context_",
                str(iteration_details.num_ctx_requests),
                "(",
                str(iteration_details.num_ctx_tokens),
                ")_generation_",
                str(iteration_details.num_generation_requests),
                "(",
                str(iteration_details.num_generation_tokens),
                ")",
            ]
        )
        return self.profiler.annotate_context_manager(annotation)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        all_gather_tensors = {}
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config

        if (
            parallel_config.pipeline_parallel_size > 1
            and compilation_config.pass_config.enable_sp
            and forward_pass
        ):
            # currently only supported by V1 GPUModelRunner
            assert not self.use_v2_model_runner
            num_scheduled_tokens_np = np.array(
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            # TODO(lucas): This is pretty gross; ideally we should only ever call
            # `_determine_batch_execution_and_padding` once (will get called again
            # in `execute_model`) but this requires a larger refactor of PP.
            _, batch_desc, _, _, _ = (
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,
                    num_reqs=len(num_scheduled_tokens_np),
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                    use_cascade_attn=False,  # TODO(lucas): Handle cascade attention
                )
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, batch_desc.num_tokens
                )
            }

        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict = get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )
            assert tensor_dict is not None
            intermediate_tensors = IntermediateTensors(tensor_dict)

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        get_pp_group().send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    # ------------------------------------------------------------------ #
    # Elastic KV: expert → KV page conversion
    # ------------------------------------------------------------------ #

    def init_elastic_kv(self) -> None:
        """Initialize elastic KV infrastructure.

        Called from engine BEFORE KV cache init. Wires VMMPagePool and
        ExpertCacheManager (created by _init_expert_offloading during
        load_model) into the elastic KV 2-phase commit protocol.

        Also computes _per_tensor_block_bytes from KV cache geometry
        so that page↔block conversion is accurate.
        """
        from vllm.elastic_kv_config import ElasticKVConfig

        config = ElasticKVConfig.from_env()
        if not config.enable:
            return

        logger.info("Elastic KV: initializing expert → KV conversion")

        # Set elastic KV flag on model_runner (enables VMM tensor path)
        self.model_runner._elastic_kv_enabled = True
        self._elastic_kv_config = config

        # Wire references from model_runner (set by _init_expert_offloading)
        self._vmm_pool = self.model_runner._vmm_pool
        self._expert_cache = self.model_runner._expert_cache

        if self._vmm_pool is None:
            logger.warning(
                "Elastic KV enabled but _vmm_pool is None. "
                "Expert offloading may not be initialized "
                "(need VLLM_EXPERT_OFFLOAD_ENABLE=1 or "
                "VLLM_VMM_EXPERT_POOL=1).")
            self.model_runner._elastic_kv_enabled = False
            self._elastic_kv_config = None
            return

        # Fail-fast: verify that shrink is actually possible.
        # Without this, prepare proposes groups but commit refuses
        # to evict → assert failure at runtime.
        if self._expert_cache is not None:
            if not self._expert_cache._can_shrink():
                logger.error(
                    "Elastic KV: expert cache cannot shrink "
                    "(phase_c=%s, max_resident=%d, local_experts=%d). "
                    "Expand will be disabled. "
                    "Set VLLM_VMM_PHASE_C=1 or reduce "
                    "VLLM_EXPERT_MAX_RESIDENT.",
                    self._expert_cache._phase_c_enabled,
                    self._expert_cache.max_resident,
                    self._expert_cache.local_num_experts,
                )
                self.model_runner._elastic_kv_enabled = False
                self._elastic_kv_config = None
                config.enable = False
                return

        # Compute per_tensor_block_bytes from KV cache spec.
        # This is called before KV cache init, so we estimate from
        # model config. The exact values will be available after
        # _initialize_kv_caches, but we need an estimate for
        # compute_derived(). We use the model's head geometry.
        per_tensor = self._compute_per_tensor_block_bytes()
        self._per_tensor_block_bytes = per_tensor
        self.model_runner._per_tensor_block_bytes = per_tensor

        # Compute derived config from VMM geometry if pool is ready
        if self._vmm_pool is not None and per_tensor:
            config.compute_derived(self._vmm_pool, per_tensor)

        # Inject Ce runtime params from expert cache / model config
        if self._expert_cache is not None:
            config.local_num_experts = getattr(
                self._expert_cache, 'local_num_experts', 0)
        if self._vmm_pool is not None:
            config.expert_group_size = getattr(
                self._vmm_pool, 'group_size', 0)
        # Extract top_k and num_layers from MoE layers if available
        moe_layers = getattr(self.model_runner, '_expert_cache_layers', None)
        if moe_layers:
            first_layer = moe_layers[0]
            config.expert_top_k = getattr(first_layer, 'top_k', 0)
            config.num_layers = len(moe_layers)

        self._elastic_kv_config = config
        logger.info(
            "Elastic KV Ce params: E=%d, G=%d, top_k=%d, L=%d",
            config.local_num_experts, config.expert_group_size,
            config.expert_top_k, config.num_layers,
        )

    def _compute_per_tensor_block_bytes(self) -> dict[int, int]:
        """Compute bytes per KV block for each KV tensor index.

        Each KV tensor stores [num_blocks, block_size, num_heads, head_size]
        for one attention layer. The per-block bytes = block_size * num_heads
        * head_size * dtype_size * 2 (K+V).
        """
        model_config = self.vllm_config.model_config
        cache_config = self.vllm_config.cache_config

        block_size = cache_config.block_size

        # Get number of KV heads and head size from model config
        hf_config = model_config.hf_config
        num_kv_heads = getattr(hf_config, 'num_key_value_heads',
                               getattr(hf_config, 'num_attention_heads', 1))
        head_dim = getattr(hf_config, 'head_dim',
                           getattr(hf_config, 'hidden_size', 4096)
                           // getattr(hf_config, 'num_attention_heads', 1))
        num_layers = getattr(hf_config, 'num_hidden_layers', 1)

        # Account for TP sharding
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        kv_heads_per_rank = max(1, num_kv_heads // tp_size)

        dtype_size = model_config.dtype.itemsize if hasattr(
            model_config.dtype, 'itemsize') else 2  # default bf16

        # Per block: block_size tokens × kv_heads × head_dim × dtype × 2(K+V)
        per_block_bytes = (
            block_size * kv_heads_per_rank * head_dim * dtype_size * 2
        )

        # One entry per attention layer
        return {i: per_block_bytes for i in range(num_layers)}

    def get_elastic_kv_config(self):
        """Return computed ElasticKVConfig for engine to pass to scheduler."""
        return getattr(self, '_elastic_kv_config', None)

    def update_elastic_kv_geometry(self, kv_cache_config) -> None:
        """Replace HF-estimated per_tensor_block_bytes with actual values.

        Called after _initialize_kv_caches() provides real KVCacheTensor sizes.
        per_block = KVCacheTensor.size // num_blocks (exact, no estimation).
        """
        if not getattr(self, '_elastic_kv_config', None):
            return

        num_blocks = kv_cache_config.num_blocks
        if num_blocks <= 0:
            return

        per_tensor = {
            idx: t.size // num_blocks
            for idx, t in enumerate(kv_cache_config.kv_cache_tensors)
        }

        old = self._per_tensor_block_bytes
        self._per_tensor_block_bytes = per_tensor
        self.model_runner._per_tensor_block_bytes = per_tensor

        # Re-run compute_derived with exact geometry
        if self._vmm_pool is not None and per_tensor:
            self._elastic_kv_config.compute_derived(
                self._vmm_pool, per_tensor)

        logger.info(
            "Elastic KV geometry updated: %d tensors, "
            "old=%s, new=%s",
            len(per_tensor),
            {k: v for k, v in list(old.items())[:3]} if old else None,
            {k: v for k, v in list(per_tensor.items())[:3]},
        )

    def elastic_kv_prepare(
        self, min_blocks: int, max_blocks: int
    ) -> dict:
        """Phase 1: report possible expansion without side effects.

        Dynamic groups: calculates required expert groups from min_blocks
        instead of using a fixed eviction count (groups_per_expand).

        The engine collects proposals from all ranks and takes the min.

        Args:
            min_blocks: Minimum KV blocks the scheduler needs.
            max_blocks: Maximum KV blocks the scheduler could use.

        Returns:
            Dict with "possible_blocks" and "groups_to_evict".
        """
        import math
        from vllm.vmm_pool import max_blocks_for_pages, pages_for_blocks

        pool = self._vmm_pool
        cache = self._expert_cache
        per_tensor = self._per_tensor_block_bytes
        cfg = self._elastic_kv_config

        if pool is None or cache is None:
            return {"possible_blocks": 0, "groups_to_evict": 0}

        free_pages = pool.num_free_pages

        # Dynamic groups: compute pages needed for min_blocks
        pages_for_min = pages_for_blocks(
            min_blocks, per_tensor, pool.page_size)
        pages_needed = max(0, pages_for_min - free_pages)

        if pages_needed == 0:
            # Free pages alone are sufficient — no eviction needed
            possible = max_blocks_for_pages(
                free_pages, per_tensor, pool.page_size)
            possible = min(possible, max_blocks)
            return {"possible_blocks": possible, "groups_to_evict": 0}

        # Calculate expert groups to evict
        group_pages = pool.group_pages
        groups_needed = math.ceil(pages_needed / group_pages)
        # Round up to quantum (min_expand_unit) for efficiency
        quantum = cfg.min_expand_unit
        groups_planned = ((groups_needed + quantum - 1) // quantum) * quantum

        evictable_groups = cache.count_evictable_groups()
        groups_to_evict = min(groups_planned, evictable_groups)

        evict_pages = groups_to_evict * group_pages
        total_pages = free_pages + evict_pages
        possible = max_blocks_for_pages(
            total_pages, per_tensor, pool.page_size)
        possible = min(possible, max_blocks)

        return {"possible_blocks": possible, "groups_to_evict": groups_to_evict}

    def elastic_kv_commit(self, agreed_blocks: int) -> dict:
        """Phase 2: execute agreed expansion (shrink experts + map KV).

        All ranks MUST expand exactly agreed_blocks or 0 (no partial).
        Partial would cause rank divergence: some ranks map more physical
        pages than the scheduler tracks as logical blocks → memory leak.
        """
        from vllm.vmm_pool import max_blocks_for_pages, pages_for_blocks

        pool = self._vmm_pool
        cache = self._expert_cache
        per_tensor = self._per_tensor_block_bytes

        self._last_commit_added = 0  # track for rollback

        if agreed_blocks <= 0 or pool is None or cache is None:
            return {"added_blocks": 0, "freed_pages": 0, "groups_evicted": 0}

        # How many pages do we need for agreed_blocks?
        pages_needed = pages_for_blocks(agreed_blocks, per_tensor, pool.page_size)
        free_pages = pool.num_free_pages
        groups_evicted = 0
        freed_pages = 0

        if free_pages < pages_needed:
            # Evict experts to free pages
            shortfall = pages_needed - free_pages
            freed_pages, groups_evicted = cache.shrink_for_pages(shortfall)

            # Verify: evict frees exactly group_pages per group,
            # no interleaving allocation between prepare and commit.
            total_available = pool.num_free_pages
            assert total_available >= pages_needed, (
                f"elastic_kv_commit: free_pages={total_available} < "
                f"pages_needed={pages_needed} after evicting "
                f"{groups_evicted} groups ({freed_pages} pages). "
                f"This should not happen: each group frees exactly "
                f"group_pages, and no allocation runs between "
                f"prepare and commit."
            )

        # Map exactly agreed_blocks worth of KV pages
        added = pool.expand_kv_physical_pages(agreed_blocks, per_tensor)
        assert added == agreed_blocks, (
            f"expand_kv_physical_pages returned {added}, expected "
            f"{agreed_blocks}. free_pages was verified sufficient above."
        )

        self._last_commit_added = added
        return {"added_blocks": added, "freed_pages": freed_pages,
                "groups_evicted": groups_evicted}

    def elastic_kv_rollback(self, n_blocks: int) -> dict:
        """Undo this rank's last successful expand_kv_physical_pages.

        Called by engine when commit diverges across ranks. Each rank
        rolls back only what it actually expanded (not the argument),
        so ranks that expanded 0 are no-ops.
        """
        pool = self._vmm_pool
        per_tensor = self._per_tensor_block_bytes
        actual = getattr(self, '_last_commit_added', 0)

        if actual <= 0 or pool is None:
            return {"rolled_back": 0}

        rolled = pool.contract_kv_physical_pages(actual, per_tensor)
        self._last_commit_added = 0
        logger.info("elastic_kv_rollback: %d blocks (requested %d)",
                     rolled, n_blocks)
        return {"rolled_back": rolled}

    def _eplb_before_scale_down(self, old_ep_size: int, new_ep_size: int) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info(
                "[Elastic EP] Starting expert resharding before scaling down..."
            )
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            execute_shuffle=True,
            global_expert_loads=None,
            rank_mapping=rank_mapping,
        )
        torch.cuda.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _eplb_after_scale_up(
        self,
        old_ep_size: int,
        new_ep_size: int,
        global_expert_loads: list[torch.Tensor] | None,
    ) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding after scaling up...")
        rank_mapping = {old_ep_rank: old_ep_rank for old_ep_rank in range(old_ep_size)}
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            execute_shuffle=True,
            global_expert_loads=global_expert_loads,
            rank_mapping=rank_mapping,
        )
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _reconfigure_parallel_config(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        """
        Update parallel config with provided reconfig_request
        """
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )

    def _reconfigure_moe(
        self, old_ep_size: int, new_ep_size: int
    ) -> list[torch.Tensor] | None:
        """
        Reconfigure MoE modules with provided reconfig_request

        Return the global expert load if new_ep_size > old_ep_size,
        otherwise None
        """
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            prepare_communication_buffer_for_model,
        )
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoE,
            FusedMoEParallelConfig,
        )

        parallel_config = self.vllm_config.parallel_config

        def get_moe_modules(model: torch.nn.Module) -> list[FusedMoE]:
            return [
                module
                for module in model.modules()
                if (
                    module.__class__.__name__ == "FusedMoE"
                    or module.__class__.__name__ == "SharedFusedMoE"
                )
            ]

        def update_moe_modules(moe_modules: list[FusedMoE], num_local_experts: int):
            assert all(
                module.moe_config.num_local_experts == num_local_experts
                for module in moe_modules
            ), "All MoE modules must have the same number of experts"
            for module in moe_modules:
                module.moe_config.num_experts = num_local_experts * new_ep_size
                module.global_num_experts = module.moe_config.num_experts
                module.moe_parallel_config = FusedMoEParallelConfig.make(
                    tp_size_=get_tp_group().world_size,
                    pcp_size_=get_pcp_group().world_size,
                    dp_size_=get_dp_group().world_size,
                    vllm_parallel_config=parallel_config,
                )
                module.moe_config.moe_parallel_config = module.moe_parallel_config
            return moe_modules

        model_moe_modules = get_moe_modules(self.model_runner.model)
        num_local_experts = model_moe_modules[0].moe_config.num_local_experts

        update_moe_modules(model_moe_modules, num_local_experts)
        drafter_model = None
        if hasattr(self.model_runner, "drafter") and hasattr(
            self.model_runner.drafter, "model"
        ):
            drafter_model = self.model_runner.drafter.model
        if drafter_model is not None and is_mixture_of_experts(drafter_model):
            drafter_moe_modules = get_moe_modules(drafter_model)
            # Check if drafter and model have matching configs
            assert (
                drafter_moe_modules[0].moe_config.num_local_experts == num_local_experts
            ), "Drafter and model configs should be the same"
            update_moe_modules(drafter_moe_modules, num_local_experts)

        if new_ep_size < old_ep_size:
            num_local_physical_experts = num_local_experts
            assert self.model_runner.eplb_state is not None
            new_physical_experts = (
                self.model_runner.eplb_state.physical_to_logical_map.shape[1]  # type: ignore[attr-defined]
            )
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts
                - self.model_runner.eplb_state.logical_replica_count.shape[1]  # type: ignore[attr-defined]
            )
            global_expert_loads = None
        else:
            num_local_physical_experts_tensor = torch.tensor(
                [num_local_experts], dtype=torch.int32, device="cpu"
            )
            torch.distributed.broadcast(
                num_local_physical_experts_tensor,
                group=get_ep_group().cpu_group,
                group_src=0,
            )
            num_local_physical_experts = int(num_local_physical_experts_tensor.item())
            new_physical_experts = num_local_physical_experts * new_ep_size
            assert self.model_runner.eplb_state is not None
            global_expert_loads_any = self.model_runner.eplb_state.rearrange(
                execute_shuffle=False
            )
            global_expert_loads = cast(list[torch.Tensor], global_expert_loads_any)
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts - global_expert_loads[0].shape[1]
            )
        prepare_communication_buffer_for_model(self.model_runner.model)
        if drafter_model is not None:
            prepare_communication_buffer_for_model(drafter_model)
        self.model_runner.model.update_physical_experts_metadata(
            num_physical_experts=new_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )
        return global_expert_loads

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
            get_ep_group,
        )

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = (
            reconfig_request.new_data_parallel_size
            * get_tp_group().world_size
            * get_pp_group().world_size
        )
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
            )

        global_expert_loads = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_loads is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size, global_expert_loads)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )

    def shutdown(self) -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    attention_config = vllm_config.attention_config
    parallel_config = vllm_config.parallel_config
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance(attention_config.backend)
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"
    init_distributed_environment(
        parallel_config.world_size, rank, init_method, local_rank, backend
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Init ec connector here before KV caches caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
