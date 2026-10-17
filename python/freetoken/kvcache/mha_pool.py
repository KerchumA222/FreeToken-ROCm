from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_cache_dtype: str = "model",
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        if kv_cache_dtype not in ("model", "q8_0", "turboquant_4bit_nc"):
            raise ValueError(f"unsupported KV cache dtype {kv_cache_dtype!r}")
        if kv_cache_dtype in ("q8_0", "turboquant_4bit_nc"):
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"{kv_cache_dtype} KV cache requires float16 or bfloat16 model compute")
        if kv_cache_dtype == "q8_0" and head_dim % 32:
            raise ValueError(f"Q8_0 KV cache requires head_dim divisible by 32, got {head_dim}")
        if kv_cache_dtype == "turboquant_4bit_nc":
            if head_dim < 2 or head_dim & (head_dim - 1):
                raise ValueError(f"TurboQuant requires an even power-of-two head_dim, got {head_dim}")
            from freetoken.kernel.triton.turboquant import gaussian_centroids, hadamard_rotation

            self._rotation = hadamard_rotation(head_dim, device)
            self._centroids, self._midpoints = gaussian_centroids(head_dim)
            self._centroids = self._centroids.to(device=device).contiguous()
            self._midpoints = self._midpoints.to(device=device).contiguous()
        else:
            self._rotation = self._centroids = self._midpoints = None
        self.kv_cache_dtype = kv_cache_dtype
        self._compute_dtype = dtype
        self._num_storage_layers = num_storage_layers
        self._num_pages = num_pages
        self._page_size = page_size
        self._local_kv_heads = local_kv_heads
        self._head_dim = head_dim
        self._device = device
        self._allocate(num_pages)

    def _allocate(self, num_pages: int) -> None:
        self._num_pages = num_pages
        self._storage_shape = (num_pages * self._page_size, self._local_kv_heads, self._head_dim)
        self._v_zero = None
        if self.kv_cache_dtype == "model":
            self._kv_buffer = torch.empty(
                (2, self._num_storage_layers, num_pages, self._page_size, self._local_kv_heads, self._head_dim),
                device=self._device,
                dtype=self._compute_dtype,
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._k_scale = self._v_scale = None
        elif self.kv_cache_dtype == "q8_0":
            shape = (
                self._num_storage_layers, num_pages, self._page_size,
                self._local_kv_heads, self._head_dim,
            )
            scale_shape = (*shape[:-1], self._head_dim // 32)
            self._kv_buffer = None
            self._k_buffer = torch.empty(shape, device=self._device, dtype=torch.int8)
            self._v_buffer = torch.empty_like(self._k_buffer)
            self._k_scale = torch.empty(scale_shape, device=self._device, dtype=torch.float16)
            self._v_scale = torch.empty_like(self._k_scale)
        else:
            shape = (
                self._num_storage_layers, num_pages, self._page_size,
                self._local_kv_heads, self._head_dim // 2,
            )
            meta_shape = (*shape[:-1],)
            self._kv_buffer = None
            self._k_buffer = torch.empty(shape, device=self._device, dtype=torch.uint8)
            self._v_buffer = torch.empty_like(self._k_buffer)
            self._k_scale = torch.empty(meta_shape, device=self._device, dtype=torch.float16)
            self._v_scale = torch.empty_like(self._k_scale)
            self._v_zero = torch.empty_like(self._k_scale)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        device = self._device
        self._k_buffer = self._v_buffer = None
        self._k_scale = self._v_scale = self._v_zero = self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._allocate(num_pages)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        specs = [
            spec for spec in config.model_config.kv_cache_group_specs()
            if spec.num_layers and not spec.is_swa
        ]
        per_token = sum(spec_kv_bytes_per_token(spec, config) for spec in specs)
        fixed = 0
        if getattr(config, "kv_cache_dtype", "model") == "turboquant_4bit_nc":
            head_dim = specs[0].head_dim if specs else config.model_config.head_dim
            # One fixed Hadamard matrix, Gaussian codebook and centroid boundaries per pool.
            fixed = (head_dim * head_dim + 16 + 15) * 4
        return per_token * config.page_size, fixed, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        tokens = self._num_pages * self._page_size
        if self._kv_buffer is not None:
            buffers = (self._kv_buffer,)
        elif self._v_zero is not None:
            buffers = (self._k_buffer, self._v_buffer, self._k_scale, self._v_scale, self._v_zero)
        else:
            buffers = (self._k_buffer, self._v_buffer, self._k_scale, self._v_scale)
        total = sum(buf.numel() * buf.element_size() for buf in buffers)
        return int(total) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def k_scale_cache(self, index: int) -> torch.Tensor | None:
        if self._k_scale is None or self.kv_cache_dtype == "turboquant_4bit_nc":
            return None
        return self._k_scale[self._dense(index)].view(
            -1, self._local_kv_heads, self._head_dim // 32
        )

    def k_norm_cache(self, index: int) -> torch.Tensor | None:
        if self.kv_cache_dtype != "turboquant_4bit_nc" or self._k_scale is None:
            return None
        return self._k_scale[self._dense(index)].view(-1, self._local_kv_heads)

    def v_scale_cache(self, index: int) -> torch.Tensor | None:
        if self._v_scale is None:
            return None
        if self.kv_cache_dtype == "turboquant_4bit_nc":
            return self._v_scale[self._dense(index)].view(-1, self._local_kv_heads)
        return self._v_scale[self._dense(index)].view(
            -1, self._local_kv_heads, self._head_dim // 32
        )

    def v_zero_cache(self, index: int) -> torch.Tensor | None:
        if self._v_zero is None:
            return None
        return self._v_zero[self._dense(index)].view(-1, self._local_kv_heads)

    @property
    def turboquant_rotation(self) -> torch.Tensor | None:
        return self._rotation

    @property
    def turboquant_centroids(self) -> torch.Tensor | None:
        return self._centroids

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        if self.kv_cache_dtype == "turboquant_4bit_nc":
            # Standard attention modules pass flattened K/V after RoPE; reshape
            # them to the head-major layout used by the TurboQuant writer.
            if (
                k.ndim == 2
                and v.shape == k.shape
                and k.shape[1] == self._local_kv_heads * self._head_dim
            ):
                k = k.reshape(k.shape[0], self._local_kv_heads, self._head_dim)
                v = v.reshape(v.shape[0], self._local_kv_heads, self._head_dim)
            from freetoken.kernel.triton.turboquant import store_turboquant_4bit_nc_cache

            assert (
                self._k_scale is not None
                and self._v_scale is not None
                and self._v_zero is not None
                and self._rotation is not None
                and self._centroids is not None
                and self._midpoints is not None
            )
            store_turboquant_4bit_nc_cache(
                k_cache=self._k_buffer[dense].view(
                    -1, self._local_kv_heads, self._head_dim // 2
                ),
                v_cache=self._v_buffer[dense].view(
                    -1, self._local_kv_heads, self._head_dim // 2
                ),
                k_norms=self._k_scale[dense].view(-1, self._local_kv_heads),
                v_scales=self._v_scale[dense].view(-1, self._local_kv_heads),
                v_zeros=self._v_zero[dense].view(-1, self._local_kv_heads),
                indices=out_loc,
                k=k,
                v=v,
                rotation=self._rotation,
                centroids=self._centroids,
                midpoints=self._midpoints,
            )
            return
        if self.kv_cache_dtype == "q8_0":
            from freetoken.kernel.triton.kv_cache import store_q8_0_cache

            store_q8_0_cache(
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                k_scale=self._k_scale[dense].view(
                    -1, self._local_kv_heads, self._head_dim // 32
                ),
                v_scale=self._v_scale[dense].view(
                    -1, self._local_kv_heads, self._head_dim // 32
                ),
                indices=out_loc,
                k=k,
                v=v,
            )
            return

        from freetoken.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._k_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
