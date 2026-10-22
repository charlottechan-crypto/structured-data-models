# Examples

This folder contains runnable examples of `structured-data-models`:

- [**`tabular/`**](tabular/): Tabular foundation model examples
- [**`relational/`**](relational/): Relational foundation model examples
- [**`timeseries/`**](timeseries/): Time series foundation model examples

See [`tabular/quickstart.py`](tabular/quickstart.py) for a minimal runnable `structured-data-models` example with `TabICLv2`.

For opt-in KumoTabular context sharding across CUDA devices, run `torchrun --standalone --nproc-per-node=2 --max-restarts=0 examples/tabular/context_sharding.py --precision float32`. Select `--precision float16` for experimental FP16 neural execution; weights and numerical inputs remain FP32. The [context-sharding example](tabular/context_sharding.py) demonstrates held-out synthetic regression queries, original query batches and clear/refit/close; see the [scope, memory options and measured scaling](../docs/source/kumotabular-context-sharding.md). Pretrained weights must be available or will be downloaded when the example runs. Use a finite external job timeout.
