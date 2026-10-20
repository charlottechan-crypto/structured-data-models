# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned tensor and control collectives."""

from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist
from torch import Tensor

T = TypeVar("T")


class Exchange:
    """Keep allocation/compute failures ahead of the next tensor collective."""

    def __init__(
        self, group: dist.ProcessGroup, control_group: dist.ProcessGroup
    ) -> None:
        self.group, self.control_group = group, control_group
        self._failure: BaseException | None = None
        self.rank, self.world = (
            dist.get_rank(group),
            dist.get_world_size(group),
        )
        self.members = tuple(dist.get_process_group_ranks(group))
        if self.members != tuple(dist.get_process_group_ranks(control_group)):
            raise ValueError("Tensor and control group membership must match")
        if dist.get_backend(control_group) != "gloo":
            raise ValueError("The control group must use Gloo")

    def records(self, value: T) -> list[T]:
        return self._records(value)

    def _records(
        self, value: T, error: BaseException | None = None
    ) -> list[T]:
        if self._failure is not None:
            raise self._failure
        values: list[Any] = [None] * self.world
        record = (None if error is None else str(error), value)
        try:
            dist.all_gather_object(values, record, group=self.control_group)
        except BaseException as collective_error:
            self._failure = error if error is not None else collective_error
            if error is None:
                raise
            raise error from collective_error
        failures = [message for message, _ in values if message is not None]
        if failures:
            self._failure = (
                error if error is not None else RuntimeError(failures[0])
            )
            raise self._failure
        return [item for _, item in values]

    def agree(self, error: BaseException | None) -> None:
        # An inner allocation failure can meet a peer's planning record.
        # Every control phase consumes the same envelope, then permanently
        # stops outer attempt frames from issuing additional collectives.
        self._records(None, error)

    def attempt(self, operation: Callable[[], T]) -> T:
        error, value = None, None
        try:
            value = operation()
        except BaseException as caught:  # noqa: BLE001
            error = caught
        self.agree(error)
        return cast(T, value)

    def metadata(self, value: T | None) -> T:
        values = [value]
        dist.broadcast_object_list(
            values, src=self.members[0], group=self.control_group
        )
        return cast(T, values[0])

    def broadcast(self, value: Tensor, owner: int = 0) -> Tensor:
        if self.world > 1:
            dist.broadcast(value, src=self.members[owner], group=self.group)
        return value

    def gather(
        self, local: Tensor, lengths: tuple[int, ...], dim: int
    ) -> Tensor:
        if self.world == 1:
            return local.clone(memory_format=torch.contiguous_format)
        dim %= local.ndim
        shape = list(local.shape)
        shape[dim] = max(lengths)
        padded = self.attempt(lambda: local.new_zeros(shape))
        padded.narrow(dim, 0, local.size(dim)).copy_(local)
        received = self.attempt(
            lambda: [torch.empty_like(padded) for _ in lengths]
        )
        dist.all_gather(received, padded, group=self.group)
        return self.attempt(
            lambda: torch.cat(
                [
                    part.narrow(dim, 0, size)
                    for part, size in zip(received, lengths)
                ],
                dim=dim,
            )
        )

    def exchange(
        self, send: Tensor, sent: list[int], received: list[int]
    ) -> Tensor:
        result = self.attempt(lambda: send.new_empty(sum(received)))
        if self.world == 1:
            result.copy_(send)
        else:
            dist.all_to_all_single(
                result,
                send,
                output_split_sizes=received,
                input_split_sizes=sent,
                group=self.group,
            )
        return result
