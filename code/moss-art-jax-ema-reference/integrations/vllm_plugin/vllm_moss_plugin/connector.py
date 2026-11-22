"""MOSS hidden-state export using the allocated cache's physical block size.

Hybrid models may split a logical cache block into smaller kernel blocks.
vLLM's pending block IDs address those kernel blocks, so using the global
logical size to expand them reads different (often untouched) cache slots.
All scheduling, transfers, locking and file serialization remain inherited
from the official connector.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
)


class MossHiddenStatesConnector(ExampleHiddenStatesConnector):
    """Use actual hidden-cache geometry for the worker's export offsets."""

    def register_kv_caches(self, kv_caches):
        from vllm.distributed import get_pp_group

        if not get_pp_group().is_last_rank:
            # Only the last PP worker owns CacheOnlyAttentionLayer. Earlier
            # stages still initialize their ordinary target KV/Mamba caches,
            # but must neither bind an absent hidden cache nor acquire the
            # final writer's filesystem locks. Inherited writer/lock methods
            # already honor the TP-writer flag; mark this stage passive too.
            self.cache_layers = []
            self._kv_cache = None
            self._is_tp_rank_zero = False
            self._moss_passive_pp_stage = True
            return
        super().register_kv_caches(kv_caches)
        self._moss_passive_pp_stage = False
        cache = self._kv_cache
        if cache is None or cache.ndim != 4:
            shape = None if cache is None else tuple(cache.shape)
            raise ValueError(
                "MOSS hidden-state export requires a four-dimensional cache "
                f"[blocks, block_tokens, layers, hidden]; received {shape}."
            )
        block_size = int(cache.shape[1])
        if block_size <= 0:
            raise ValueError("MOSS hidden-state cache block size must be positive")
        # Only the worker has a bound cache and performs _submit_async_write.
        # Do not change the shared CacheConfig or scheduler's logical block size.
        self._block_size = block_size
