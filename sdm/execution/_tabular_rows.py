# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original global cell/column/row windows with contiguous row ownership."""

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
from torch import Tensor

from sdm._memory import chunk_memory_limit
from sdm.cache import Cache
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_columns import _column_forward, domain_rounds
from sdm.models.kumo.tabular.row_embedding import RowEmbedding
from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.nn import InducedTransformerBlock, TransformerBlock


def bounds(total: int, world: int) -> tuple[int, ...]:
    return tuple(
        total * (rank + 1) // world - total * rank // world
        for rank in range(world)
    )


class Rows:
    """Bind native AUTO plans once, before allocating expanded cells."""

    def __init__(
        self,
        encoder: RowEmbedding,
        raw: Tensor,
        labels: Tensor,
        mask: Tensor,
        exchange: Exchange,
        staging_bytes: int,
    ) -> None:
        self.encoder, self.raw, self.labels = encoder, raw, labels
        self.exchange, self.mask = exchange, mask
        self.total, self.columns = raw.shape
        self.lengths = bounds(self.total, exchange.world)
        self.start = sum(self.lengths[: exchange.rank])
        self.stop = self.start + self.lengths[exchange.rank]
        self.readouts, self.channels = encoder.readout_token.shape
        self.mean = raw.nanmean(dim=-2, keepdim=True)
        self.mean = self.mean.masked_fill(self.mean.isnan(), 0)
        self.cell_rows = self.total
        cell = encoder.cell_embedding
        if raw.is_cuda:
            groups, frequencies = cell.group_size, cell.num_frequencies
            fixed = self.columns * (
                groups * frequencies * 4
                + groups * self.channels * 2 * frequencies * 4
                + self.channels * 4
            )
            item = (2 * groups * frequencies + 2 * self.channels) * 4
            items = max((chunk_memory_limit(raw.device) - fixed) // item, 1)
            self.cell_rows = max(items // self.columns, 1)
        self.handlers: dict[torch.nn.Module, Callable[..., Any]] = {
            encoder.cell_embedding: self.cell,
            encoder.norm: self.norm,
        }
        if encoder.y_lin is not None:
            self.handlers[encoder.y_lin] = self.target
        if encoder.y_emb is not None:
            self.handlers[encoder.y_emb] = encoder.y_emb
        limits = []
        peak = 0
        for module in encoder.col_blocks:
            column = cast(InducedTransformerBlock, module)
            points = column.inducing_points.size(0)
            iw = min(
                column.inducing_block.auto_batch_size_limit(
                    raw.device, 4, points, self.total
                ),
                65_535,
            )
            ow = min(
                column.output_block.auto_batch_size_limit(
                    raw.device, 4, self.total, points
                ),
                65_535,
            )
            limits.append((iw, ow))
            for domains in domain_rounds(self.columns, iw, ow, exchange.world):
                width = sum(d.stop - d.start for d in domains)
                for peer, length in enumerate(self.lengths):
                    owned = sum(
                        d.stop - d.start for d in domains if d.owner == peer
                    )
                    sent = length * width * self.channels * 4
                    received = self.total * owned * self.channels * 4
                    size = max(2 * (sent + received), 3 * received)
                    if (self.columns <= iw or self.columns <= ow) and owned:
                        size += (
                            self.total
                            * (self.columns + self.readouts)
                            * self.channels
                            * 4
                        )
                    peak = max(peak, size)
            self.handlers[column] = _column_forward(
                column,
                column.inducing_block,
                column.output_block,
                lengths=self.lengths,
                group=exchange,
                readouts=self.readouts,
            )
        row_limits = []
        for index, module in enumerate(encoder.row_blocks):
            block = cast(TransformerBlock, module)
            last = index == len(encoder.row_blocks) - 1
            tokens = self.readouts if last else self.columns + self.readouts
            limit = min(
                block.auto_batch_size_limit(
                    raw.device, 4, tokens, self.columns + self.readouts
                ),
                65_535,
            )
            row_limits.append(limit)
            self.handlers[block] = partial(self.row, block, limit, tokens)
        if peak > staging_bytes:
            raise ValueError(
                "Column staging budget is too small for native windows"
            )
        plan = (self.cell_rows, limits, row_limits)
        if any(other != plan for other in exchange.records(plan)):
            raise ValueError("Native AUTO windows differ between owners")

    def __call__(
        self, module: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> Any:
        return self.handlers[module](*args, **kwargs)

    def cell(
        self,
        *,
        x: Tensor,
        categorical_mask: Tensor,
        cache: Cache,
        out: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        del x, kwargs
        cache["cell_impute_mean"] = self.mean
        for index, start in enumerate(range(0, self.total, self.cell_rows)):
            stop = min(start + self.cell_rows, self.total)
            self._cell_window(start, stop, index, categorical_mask, out)
        return out

    def _cell_window(
        self, start: int, stop: int, index: int, mask: Tensor, out: Tensor
    ) -> None:
        owner = self.owner(start, stop)
        interior = owner is not None
        owner = index % self.exchange.world if owner is None else owner

        def allocate() -> Tensor | None:
            if interior and self.exchange.rank != owner:
                return None
            buffer = self.raw.new_empty(
                stop - start, self.readouts + self.columns, self.channels
            )
            buffer[:, : self.readouts] = self.encoder.readout_token
            return buffer

        buffer = self.exchange.attempt(allocate)

        def compute() -> None:
            if self.exchange.rank == owner:
                assert buffer is not None
                raw = self.raw[start:stop]
                missing = raw.isnan()
                CellEmbedding.forward(
                    self.encoder.cell_embedding,
                    torch.where(missing, self.mean, raw),
                    mask,
                    batch_size_limit=None,
                    out=buffer[:, self.readouts :],
                    missing=missing,
                )

        self.exchange.attempt(compute)
        if not interior:
            assert buffer is not None
            self.exchange.broadcast(buffer, owner)
        left, right = max(start, self.start), min(stop, self.stop)
        if left < right:
            assert buffer is not None
            out[left - self.start : right - self.start].copy_(
                buffer[left - start : right - start, self.readouts :]
            )

    def owner(self, start: int, stop: int) -> int | None:
        offset = 0
        for rank, length in enumerate(self.lengths):
            if offset <= start < stop <= offset + length:
                return rank
            offset += length
        return None

    def target(self, labels: Tensor) -> Tensor:
        del labels
        assert self.encoder.y_lin is not None
        return self.encoder.y_lin(self.labels.unsqueeze(-1))[
            self.start : self.stop
        ].clone()

    def row(
        self,
        block: torch.nn.Module,
        limit: int,
        tokens: int,
        *,
        key_value: Tensor,
        out: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        interior, boundary = [], []
        for index, start in enumerate(range(0, self.total, limit)):
            stop = min(start + limit, self.total)
            owner = self.owner(start, stop)
            if owner is None:
                boundary.append((index, start, stop))
            elif owner == self.exchange.rank:
                interior.append((start, stop))

        def local_interiors() -> None:
            # Preserve each native window's shape and clone layout, but avoid
            # a control collective for every rank-local window. Rows in
            # different windows are independent in this attention block.
            for start, stop in interior:
                full = key_value[
                    start - self.start : stop - self.start
                ].clone()
                query = full[:, :tokens]
                block(
                    query=query,
                    key_value=full,
                    out=query,
                    batch_size_limit=limit,
                )
                out[start - self.start : stop - self.start].copy_(
                    full[:, :tokens]
                )
                del query, full

        self.exchange.attempt(local_interiors)
        # Boundary collectives start only after all owners have completed
        # their local enqueue phase or agreed on its first error.
        for index, start, stop in boundary:
            self._row_window(
                block, limit, tokens, key_value, out, index, start, stop
            )
        return out

    def _row_window(
        self,
        block: torch.nn.Module,
        limit: int,
        tokens: int,
        key_value: Tensor,
        out: Tensor,
        index: int,
        start: int,
        stop: int,
    ) -> None:
        owner = self.owner(start, stop)
        interior = owner is not None
        owner = index % self.exchange.world if owner is None else owner
        left, right = max(start, self.start), min(stop, self.stop)
        count = max(right - left, 0)
        local = key_value[left - self.start : left - self.start + count]
        if interior:
            full = self.exchange.attempt(
                lambda: local.clone() if self.exchange.rank == owner else None
            )
        else:
            lengths = tuple(
                max(
                    min(stop, sum(self.lengths[: r + 1]))
                    - max(start, sum(self.lengths[:r])),
                    0,
                )
                for r in range(self.exchange.world)
            )
            sent = [0] * self.exchange.world
            sent[owner] = local.numel()
            received = [
                n * (self.columns + self.readouts) * self.channels
                if self.exchange.rank == owner
                else 0
                for n in lengths
            ]
            gathered = self.exchange.exchange(
                local.contiguous().view(-1), sent, received
            )
            full = self.exchange.attempt(
                lambda: key_value.new_empty(
                    stop - start, self.columns + self.readouts, self.channels
                )
            )
            if self.exchange.rank == owner:
                full.copy_(gathered.view_as(full))

        def compute() -> None:
            if self.exchange.rank == owner:
                assert full is not None
                query = full[:, :tokens]
                block(
                    query=query,
                    key_value=full,
                    out=query,
                    batch_size_limit=limit,
                )

        self.exchange.attempt(compute)
        if not interior:
            assert full is not None
            self.exchange.broadcast(full, owner)
        if count:
            assert full is not None
            out[left - self.start : left - self.start + count].copy_(
                full[left - start : left - start + count, :tokens]
            )

    def norm(self, value: Tensor) -> Tensor:
        full = self.exchange.gather(value, self.lengths, 0)
        return self.encoder.norm(full)[self.start : self.stop].clone()
