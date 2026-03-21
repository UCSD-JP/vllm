# SPDX-License-Identifier: Apache-2.0
"""Cross-layer expert routing prediction for prefetching.

All expert IDs are LOCAL (EP shard-local).
Uses vLLM's expert_map for global->local conversion.
expert_map=None (ep_size=1) safe: global_id == local_id.
"""

import torch
from collections import defaultdict
from typing import Dict, List, Optional


class ExpertPredictor:
    """Predict next layer's active experts from current layer's routing."""

    def __init__(
        self,
        num_layers: int,
        num_local_experts: int,
        top_k: int,
    ):
        self.num_layers = num_layers
        self.num_local_experts = num_local_experts
        self.top_k = top_k

        self._freq_counter: Dict[int, Dict[int, int]] = {
            l: defaultdict(int) for l in range(num_layers)
        }
        self._freq_topk: Dict[int, List[int]] = {}
        self._update_interval = 100
        self._step = 0

    def update_and_predict(
        self,
        current_layer_idx: int,
        topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor],
        target_layer_idx: int,
    ) -> List[int]:
        """Update stats + predict next layer experts. Returns local IDs."""
        # Global -> local conversion
        if expert_map is not None:
            emap_cpu = expert_map.cpu()
            local_ids = set()
            for gid in topk_ids.flatten().unique().tolist():
                if 0 <= gid < emap_cpu.shape[0]:
                    lid = emap_cpu[gid].item()
                    if lid != -1:
                        local_ids.add(lid)
        else:
            # ep_size=1: global_id == local_id
            local_ids = set(topk_ids.flatten().unique().tolist())

        # Update frequency stats
        for lid in local_ids:
            self._freq_counter[current_layer_idx][lid] += 1
        self._step += 1
        if self._step % self._update_interval == 0:
            self._refresh_topk()

        # Predict: same-expert heuristic + frequency fill
        predicted = list(local_ids)
        if len(predicted) < self.top_k:
            freq_top = self._freq_topk.get(target_layer_idx, [])
            seen = set(predicted)
            for eid in freq_top:
                if eid not in seen:
                    predicted.append(eid)
                    if len(predicted) >= self.top_k:
                        break
        return predicted[:self.top_k]

    def _refresh_topk(self):
        for l in range(self.num_layers):
            c = self._freq_counter[l]
            if c:
                self._freq_topk[l] = [
                    e for e, _ in sorted(c.items(), key=lambda x: -x[1])
                ][:self.top_k * 3]
