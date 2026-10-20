# KumoTabular context sharding

`sdm.execution.tabular_execution` is an experimental, opt-in executor for splitting one KumoTabular estimator's context across devices. Ordinary `model.fit` and `model.predict` retain their existing behavior when no executor owns the model.

## Usage

Launch one process per CUDA device with `torchrun --standalone --nproc-per-node=4 your_script.py`. Each process loads identical weights and creates the same ordered tensor and control groups:

```python
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from sdm.execution import tabular_execution
from sdm.models import KumoTabular

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(minutes=10))
control = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))
root = dist.get_rank() == 0
model = KumoTabular(task="regression", device=f"cuda:{local_rank}").eval()

run = tabular_execution(
    model,
    task="regression",
    group=dist.group.WORLD,
    control_group=control,
    fit_head_gather_window=2,
)
# Only root supplies tensors. All ranks call every method in the same order.
# x_context: [context_rows, features]; y_context: [context_rows, 1].
run.fit(x_context if root else None, y_context if root else None)
prediction = run.predict(x_query if root else None)
# A list of original batches allows independent batches to run concurrently.
predictions = run.predict_batches(query_batches if root else None)
run.clear()
run.fit(x_context if root else None, y_context if root else None)
run.close()
# The caller owns and destroys groups after healthy completion.
dist.destroy_process_group(control)
dist.destroy_process_group()
```

Inputs may be tensors or `TableTensor`s. Root alone supplies the recipe and optional preprocessing generator. Prediction returns the usual public `TableTensor` on root and `None` on peers; `predict_batches` returns a root-only list in the original order. Each original query batch is kept intact, including recipe output processing and regression inverse transforms. Empty batch lists return an empty list.

## What is distributed

- Expanded row-embedding cells are stored by contiguous context-row owner, reducing the dominant row-by-feature-by-channel allocation.
- Column attention exchanges row-owned cells into complete native column windows, then returns results to their row owners. It does not replace full-context column attention with independent local fits.
- Independent row-attention windows run on their owner, with one error agreement for the local loop. Windows crossing owner boundaries retain their native shape and arithmetic.
- ICL fit attention heads are divided across owners. Full projections and residuals remain replicated; retained ICL key/value caches are sharded by context row.
- Prediction gathers one cache layer at a time and assigns complete original query batches to owners. This combines context-owned storage with query-batch parallelism, not ensemble parallelism.

This is not complete tensor parallelism and does not guarantee linear capacity or speed scaling. Raw prepared inputs, some normalization/projection intermediates, full ICL states, one replayed cache layer, and a native column window can still scale with the entire context on a single device. Communication and replicated work can dominate small inputs. `column_staging_bytes` and `attention_workspace_bytes` bound individual planned components, not total allocated, reserved, or physical GPU memory.

## Supported scope and lifecycle

The initial scope is eager evaluation, FP32, one unbatched estimator, stock KumoTabular classification or regression, and no callbacks, custom forward hooks, compilation, autocast, or ECOC expansion beyond native class capacity. The number of owners must divide the ICL fit query/key-value head counts. CPU/Gloo is supported for numerical and lifecycle tests; CUDA requires NCCL for tensors plus a separate Gloo control group with identical ordered members.

The executor borrows its model exclusively. Do not mutate weights, recipes, caches, execution settings, or use the underlying model while the lease is active. `fit` replaces a fitted generation; `clear` releases it while retaining the lease; healthy `close` releases the lease and permits ordinary model reuse. Calls must originate on the constructing thread and cannot overlap. Groups and process lifetime always remain caller-owned.

Local allocation or compute errors are agreed before the next tensor collective, and the executor retains its first failure. A failed executor cannot be cleared, refitted, or reopened; retire the worker processes, especially after CUDA or collective failures. A process crash or mismatched method order still requires the caller's finite process-group timeout and job supervision. Never attempt a collective `close` in a one-rank-only exception handler.

## Validation status

Tests compare complete public classification outputs and all 999 regression quantiles with the same native model on one, two, and four CPU processes. They also exercise original row-window boundaries, refit, clear, close, borrowed process groups, mutation rejection, and collective failure propagation. These CPU checks do not establish CUDA numerical equivalence, trained-model quality, memory scaling, or speedup for this branch. Real-GPU matched native/one/two/four-owner qualification remains required; historical research-prototype performance must not be treated as a measurement of this current-main implementation.
