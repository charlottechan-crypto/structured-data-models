# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owning cache slices, with one layer gathered during replay."""

from collections.abc import Hashable, Iterator
from typing import Self, cast

from torch import Tensor

from sdm.cache import Cache, KVCacheEntry
from sdm.execution._collectives import Exchange


class OwnedCache(Cache):
    """Store columns for the frontend or context rows for ICL."""

    def __init__(
        self,
        exchange: Exchange,
        lengths: tuple[int, ...],
        dim: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.exchange, self.lengths = exchange, lengths
        self.dim, self.prefix = dim, prefix
        self.start = sum(lengths[: exchange.rank])
        self.length = lengths[exchange.rank]

    def __setitem__(self, key: Hashable, value: object) -> None:
        if isinstance(key, str) and key.startswith(self.prefix):
            pair = cast(KVCacheEntry, value)
            value = KVCacheEntry(
                *(
                    tensor.narrow(self.dim, self.start, self.length)
                    .clone()
                    .contiguous()
                    for tensor in pair
                )
            )
        super().__setitem__(key, value)

    def __getitem__(self, key: Hashable) -> object:
        value = super().__getitem__(key)
        if (
            self.is_replaying
            and isinstance(key, str)
            and key.startswith(self.prefix)
        ):
            pair = cast(KVCacheEntry, value)
            return KVCacheEntry(
                *(
                    self.exchange.gather(
                        tensor, self.lengths, self.dim
                    ).contiguous()
                    for tensor in pair
                )
            )
        return value

    def _tensors(self) -> Iterator[Tensor]:
        # Iteration over Cache.values() would gather replay entries.
        for value in self._items.values():
            if isinstance(value, KVCacheEntry):
                yield from value
            elif isinstance(value, Tensor):
                yield value

    def freeze(self) -> Self:
        # Entries contain owning tensors only; do not invoke replay getters.
        self._mode = Cache.Mode.replay
        return self
