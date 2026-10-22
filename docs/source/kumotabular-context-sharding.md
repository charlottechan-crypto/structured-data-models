# KumoTabular context sharding

`sdm.execution.tabular_execution` is an experimental, opt-in executor for splitting one KumoTabular estimator's context across devices. Ordinary `model.fit` and `model.predict` retain their existing behavior when no executor owns the model.

For large contexts, the measured configuration combines **context-owned row embeddings**, **owner-local frontend computation**, and **sharded row workspaces**. FP16 neural execution is a separate, explicit option. This scales one fitted model's context; it does not train independent models on subsets or ensemble their predictions.

## Usage

Run the self-contained [synthetic regression example](../../examples/tabular/context_sharding.py) from the repository root:

```sh
torchrun --standalone --nproc-per-node=2 --max-restarts=0 examples/tabular/context_sharding.py --precision float32
```

Use `--precision float16` to exercise the FP16 path. The example loads the stock large regression checkpoint on each device (downloading weights if absent), fits preprocessing only on the context rows, predicts original held-out query batches, then clears, refits and closes. Its small synthetic input is not a performance or quality benchmark. Run under a finite external job timeout, with one process per CUDA device; the measured frontend configuration supports one, two, four or eight processes.

For integration into another script, each rank loads identical weights and creates the same ordered tensor and control groups. The following fragment assumes caller-provided context and query tensors on rank zero's CUDA device; numerical inputs and regression targets are FP32, while classification labels preserve their integer dtype:

```python
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from sdm.execution import tabular_execution
from sdm.models import KumoTabular

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group("nccl", timeout=timedelta(minutes=10))
control = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))
root = dist.get_rank() == 0
model = KumoTabular(task="regression", size="large", device=device).eval()

run = tabular_execution(
    model,
    task="regression",
    group=dist.group.WORLD,
    control_group=control,
    fit_head_gather_window=2,
    owner_local_frontend=True,
    shard_row_workspace=True,
    precision="float32",  # Explicitly choose "float16" for neural autocast.
)
# Only root supplies tensors. All ranks call every method in the same order.
# x_context: [context_rows, features]; y_context: [context_rows, 1].
with torch.no_grad():
    run.fit(x_context if root else None, y_context if root else None)
    prediction = run.predict(x_query if root else None)
    # Original query batches can run concurrently without splitting a batch.
    predictions = run.predict_batches(query_batches if root else None)
    run.clear()
    run.fit(x_context if root else None, y_context if root else None)
    run.close()
# The caller owns and destroys groups after healthy completion.
dist.destroy_process_group(control)
dist.destroy_process_group()
```

Inputs may be tensors or `TableTensor`s, with shapes `[context_rows, features]`, `[context_rows, 1]` for targets, and `[query_rows, features]`. Root alone supplies inputs, the recipe and an optional preprocessing generator; peers supply `None`. Prediction returns the usual public `TableTensor` on root and `None` on peers; `predict_batches` returns a root-only list in the original order. Each original query batch is kept intact, including recipe output processing and regression inverse transforms. Empty batch lists are supported; individual empty batches are not. `predict` does not automatically partition one query batch.

## Configuration and compatibility

The original FP32 behavior remains the default. Enable the memory-saving frontend explicitly:

| Option | Default | Purpose |
| --- | --- | --- |
| `owner_local_frontend` | `False` | Compute each owner's part of cell and row windows directly, avoiding boundary staging. |
| `shard_row_workspace` | `False` | Divide the native row-block chunk limit by the owner count; requires owner-local frontend and 1, 2, 4 or 8 owners. |
| `precision` | `"float32"` | `"float16"` enables per-worker neural autocast; weights, prepared numerical inputs and public processing remain FP32. |
| `fit_head_gather_window` | `None` | Limit outstanding fit head gathers; the measured configuration uses `2`. |
| `column_staging_bytes` | `8 << 30` | Admission budget for column exchange staging. |
| `attention_workspace_bytes` | `12 << 30` | Admission budget for layer-local prediction attention and cache transport. The large-context measurements below use `48 << 30`. |

These component budgets are not total device-memory limits. The example retains the public 12 GiB attention budget; it does not reproduce the large-context benchmark configuration automatically. Choose budgets for the workload and device. An oversized original query batch is rejected rather than silently split.

The FP16 option owns matching autocast scopes on every worker. Do not call `model.half()` or cast preprocessing inputs to FP16. Do not wrap the default FP32 executor in autocast as a substitute for selecting `precision="float16"`. BF16 and half-stored model weights are not supported. FP16 can differ from FP32; same-precision distributed/native agreement is a separate question from precision-related quality changes.

## What is distributed

- Expanded row-embedding cells are stored by contiguous context-row owner, reducing the dominant row-by-feature-by-channel allocation.
- Column attention exchanges row-owned cells into complete native column windows, then returns results to their row owners. It does not replace full-context column attention with independent local fits.
- With the default frontend, independent row-attention windows run on their owner and boundary windows retain their native shape. With owner-local frontend enabled, each rank computes its intersections of the native windows. Sharded row workspace additionally bounds the original whole-row-block chunker to `max(1, native_row_limit // owners)`. These options change projection batch geometry and require numerical validation.
- ICL fit attention heads are divided across owners. Full projections and residuals remain replicated; retained ICL key/value caches are sharded by context row.
- Prediction gathers one cache layer at a time and assigns complete original query batches to owners. This combines context-owned storage with query-batch parallelism, not ensemble parallelism.

This is not complete tensor parallelism and does not guarantee linear capacity or speed scaling. Raw prepared inputs, some normalization/projection intermediates, full ICL states, one replayed cache layer, and a native column window can still scale with the entire context on a single device. Communication and replicated work can dominate small inputs. `column_staging_bytes` and `attention_workspace_bytes` bound individual planned components, not total allocated, reserved, or physical GPU memory.

For the tested large model, the expanded row tensor has shape `[context_rows, prepared_features + 4, 256]`. At 2,097,152 rows and 500 prepared features in FP16, this tensor alone is 504 GiB without sharding, 126 GiB per rank with four balanced owners, and 63 GiB with eight. Those are component sizes, not predictions of whole-workflow peak memory. Chunk limits bound temporary work, not the total number of context rows.

## Supported scope and lifecycle

The supported scope is eager evaluation, FP32 or explicit FP16 neural execution, one unbatched estimator, stock KumoTabular classification or regression, and no autograd inputs, callbacks, custom dispatch/forward hooks, compilation, or ECOC expansion beyond native class capacity. The number of owners must divide both ICL fit query- and key/value-head counts. CPU/Gloo supports numerical and lifecycle tests; CUDA requires NCCL for tensors plus a separate Gloo control group with identical ordered members. Multi-node and heterogeneous-device execution have not been qualified.

Use normal calls or `torch.no_grad()`, **not an outer `torch.inference_mode()`**: fitted-cache identity checks require tensor version counters. Constructor, fit and prediction reject inference mode. Healthy `clear` and `close` are allowed inside inference mode.

The executor borrows its model exclusively. Do not mutate weights, recipes, caches, execution settings, or use the underlying model while the lease is active. `fit` replaces a fitted generation; `clear` releases it while retaining the lease; healthy `close` releases the lease and permits ordinary model reuse. Calls must originate on the constructing thread and cannot overlap. Groups and process lifetime always remain caller-owned.

Local allocation or compute errors are agreed before the next tensor collective, and the executor retains its first failure. A failed executor cannot be cleared, refitted, or reopened; retire the worker processes, especially after CUDA or collective failures. A process crash or mismatched method order still requires the caller's finite process-group timeout and job supervision. Never attempt a collective `close` in a one-rank-only exception handler.

## Validation status

The CPU execution tests cover complete public classification outputs and all 999 regression quantiles, one/two/four/eight-owner paths, uneven and empty owners, default-recipe RNG preservation, precision-policy agreement, row workspace bounds, refit, clear, close, borrowed groups, mutation rejection and collective failure propagation. Numerical comparisons use `atol=2e-6, rtol=1e-5`. CPU checks alone do not establish trained CUDA equivalence or performance.

During consolidation, all 252 execution tests passed on the imported measured runtime. After the FP32 backend compatibility adjustment described below, 83 focused backend-policy, precision and lifecycle tests passed on the final runtime. These sets overlap; they are not 335 distinct tests or a complete final-source suite rerun. Both runs had ten CPU Half-input/FP32-weight RMSNorm fallback warnings and no failures or skips. Ruff, formatting, diff checks and the example's CPU-only `--help` also passed. Run `python -m pytest test/execution` from the repository root for the complete execution suite.

The measured runtime identified below passed trained FP16 native-versus-eight-owner classification (EPSILON, 4,096 context rows) and all-999-quantile regression (YearPrediction, 16,384 rows) checks, plus the larger EPSILON comparison below. These are selected checkpoints and inputs, not general precision guarantees. Earlier FP16 source attempts failed strict native parity; their results are not silently included in the passing cohort.

Dedicated CUDA lifecycle diagnostics for this runtime remain pending: earlier attempts failed test collection, and the packaging/device fixture corrections were checked locally only. Older-source CUDA lifecycle passes do not qualify the successor. The trained benchmark runs do exercise repeated fit/predict/clear/close, but are not a substitute for those dedicated failure and ownership tests.

## Measured FP16 scaling

These measurements use a single eight-B200 host, PyTorch 2.9.1+cu130, cuDNN 9.2, one large classification estimator, TF32 disabled, both frontend options enabled, `fit_head_gather_window=2`, 8 GiB column staging and 48 GiB attention workspace. Each case has 512 fixed held-out validation queries in original batches of eight. Preprocessing and feature selection use training/context data only. The measured predecessor's 176-file SDM Python source aggregate is `2f5314f0efd69cad02135c9a66a502de5eadd12f2a692a388aefd7e9dab67e43`.

This branch incorporates that runtime with one compatibility adjustment: cuDNN fit attention is eligible only under CUDA FP16 autocast, preserving the original flash/efficient-only policy for default FP32 calls. The enabled backend set in the measured FP16 path is unchanged. The consolidated source has CPU tests but has not been re-run on GPUs; the following measurements belong to the identified predecessor, not a fresh qualification of this branch revision.

Each GPU count has one clean multiprocess run, with one worker per GPU, three complete synchronized fits and twelve complete predictions. Fit medians include all three fits; prediction medians use nine calls, excluding the first call after each fit. Memory includes setup, warmups, refits, clear and close on every rank. Allocated memory is the busiest rank's allocator high-water; physical memory is sampled device usage. These are not independent-run confidence intervals or full-process cold-start measurements.

### EPSILON: 262,144 context rows

The input has 2,000 raw features and 500 prepared features. The one-GPU denominator is the same executor with the same options, not a different native implementation.

| GPUs | Fit median (s) | Predict 512 rows (s) | Peak allocated / GPU (GiB) | Busiest physical (GiB) | Aligned all-device physical (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.731 | 11.610 | 77.16 | 79.89 | 79.92 |
| 2 | 11.368 | 6.185 | 41.32 | 49.24 | 95.48 |
| 4 | 6.056 | 4.134 | 25.36 | 36.30 | 130.41 |
| 8 | 3.285 | 3.647 | 17.38 | 29.93 | 214.80 |

Eight GPUs yield **6.31× fit and 3.18× prediction speedup** over one. First-fit times are 26.442 / 19.890 / 17.765 / 22.531 seconds respectively: the median speedup is not a first-fit speedup. Prediction scaling flattens at this workload, and aggregate memory increases despite lower per-device peaks.

Complete outputs are bit-identical across all four GPU counts and the native FP16 reference. Selected-512 validation accuracy is 0.765625 and log loss is 0.48790048. A historical native FP32 reference from a different host has accuracy 0.767578125 and log loss 0.48788189, with one changed label relative to FP16. This contrast is descriptive, not a matched current-host precision experiment, a quality improvement or a relaxed distributed parity gate.

### URL: 2,097,152 context rows

The input has 500 prepared features: more than one billion context cells. Native one-GPU execution runs out of memory requesting the 504 GiB expanded row buffer; the two-owner executor fails requesting 252 GiB per rank. These failed requests are not observed allocated peaks.

| GPUs | Outcome | Fit median (s) | Predict 512 rows (s) | Peak allocated / GPU (GiB) | Busiest physical (GiB) | Aligned all-device physical (GiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Native OOM | — | — | — | — | — |
| 2 | Executor OOM | — | — | — | — | — |
| 4 | Completes | 166.145 | 23.663 | 144.38 | 159.37 | 599.84 |
| 8 | Completes | 85.096 | 11.903 | 80.53 | 97.87 | 695.11 |

Four to eight GPUs gives **1.95× fit and 1.99× prediction speedup**, with 44% lower busiest-rank allocated memory. First fits are 176.208 and 103.237 seconds. All complete four- and eight-owner predictions are bit-identical; selected-512 validation accuracy is 0.974609375 and log loss is 0.07221999. There is no successful native oracle or selected FP32 quality reference at this size.

This is near-linear **fixed-context speed scaling from four to eight GPUs**, not a universal linear maximum-context law. It demonstrates a workload that fits with multiple GPUs but not one; it does not locate the largest possible context. The smaller EPSILON and larger URL cases must not be combined into a same-dataset capacity curve.

### Measurement identities

The source hash is `sha256(json.dumps(mapping, sort_keys=True).encode())`, where `mapping` contains every recursively found Python file's path relative to `sdm/` (for example, `execution/tabular.py`) and its SHA-256 hex digest. Timing and numerical evidence is tied to that source and the individual workloads, not to every future revision of this branch.

| Record | SHA-256 |
| --- | --- |
| Consolidated runtime, CPU-validated compatibility successor | `8c00b18e7996a0812c48481bdfe6ac3db4825bf38709c53828f020f098e3f320` |
| EPSILON timing runner | `d8f1b2b3e6f49a5f6cf57639c2d080b881e454fc882578e5e35cf5783aa976f8` |
| EPSILON accepted four-point comparison | `f0d5f989d73b2bddc23daed4e4f542f7fc37a6f4ca83496d85f75e3ab6581a08` |
| URL timing runner | `23a8fa5cd6e293253b0ae219fd045ee93f926d046fdad56b0d4dfc451361a3f9` |
| URL accepted four/eight-GPU comparison | `91adbbae737128724bbe023e7a056066ee1d9727216883bacf99e6eb4ca66880` |

These identify the retained research records; raw experiment bundles and dataset caches are not shipped in this library branch. The synthetic example is an API example, not a reproduction of these timings. Local-final-normalization, alternative collectives and replicated prediction-KV prototypes are not part of this implementation or its claims.

### Earlier branch measurements

The following historical results apply to the original FP32 implementation, not a matched before/after measurement of the current FP16 path. They are retained separately to avoid mixing source versions, precision or timing denominators.

Separately, published revision `131f794254dccf78e3ea396cc3523d86dfeaacc2` has trained LARGE FP32/E1 CUDA qualification on B200 GPUs. Matched EPSILON C131072, 2000 raw → 500 prepared features, Q512 in original B8 batches completes three fits and twelve predictions on native and one/two/four owners; full saved output, prepared-value, RNG, model-state and axis checks pass with zero output differences or class flips. Separate clean four-owner fit/predict medians are 22.444074/3.771376 seconds versus native 77.864580/9.322108 seconds (3.469×/2.472×). California regression C4097/F8/Q64 passes full all-999-quantile checks across native/one/two/four owners; its single query batch does not establish query-parallel throughput scaling.

The same production source also has matched EPSILON C262144 native/four/eight-owner qualification under runner `7a40a89b26ad05857f59e9f347fdd2c7436ac0e2c06ce74a4210af48cf891f81`. This workflow uses `column_staging_bytes=8 << 30`, `attention_workspace_bytes=48 << 30`, `fit_head_gather_window=2`, TF32 disabled and the stock default recipe. Clean fit/predict medians are 226.735692/18.492159 seconds native, 61.366311/5.048313 with four owners and 34.625737/3.527150 with eight. The 48-GiB workspace setting is an explicit benchmark component admission override; the example and usage snippet above use the public default of 12 GiB. These results do not establish the same capacity for a default-budget launch.

At C350000, two/four owners complete where native's first fit OOMs, and their full outputs agree. Clean two→four fit/predict medians are 187.399507→96.843257 and 13.059308→6.636829 seconds. There is no native prediction oracle or native speed denominator at that size. C131072 busiest allocated memory falls from 76.412 GiB native to 30.382 GiB on four owners, while sampled aggregate physical rises from 82.174 to 142.615 GiB. C350000 aggregate physical similarly rises from 211.832 to 252.496 GiB on two→four owners. These sequential, workload-specific measurements are not confidence intervals, universal linear scaling, maximum-context bounds or mixed-precision qualification. Fixed validation subsets are not TEST scores. Later experimental source variants and local documentation/example/test edits do not inherit qualification beyond these recorded source and workflow identities.
