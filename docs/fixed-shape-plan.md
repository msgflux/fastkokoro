# Whole-Graph Fixed-Shape Plan

The previous experiment fixed only the external token input shape and then sliced
it back to the real token length inside the graph. That preserves semantic
correctness, but it does not make the model fixed-shape in the way ONNX Runtime
and CUDA Graph care about. The dynamic `Slice` output becomes the shape seen by
the rest of the graph.

This document turns that lesson into an execution plan.

## What Has To Change

Whole-graph fixed-shape readiness requires all three of these to be true:

1. The externally visible request tensors are bucketed and fixed.
2. The internal graph path that produces the first streamed audio chunk remains
   inside a small set of stable shapes.
3. The output path also stays stable enough that ONNX Runtime does not keep
   paying per-run shape churn and allocation churn.

Changing only the input tensor shape is not enough if an early `Slice`,
`Reshape`, `Shape`, `Gather`, or `Expand` immediately reconstructs a dynamic
dimension.

## Plan

### Phase 0: Inspect The Current Graph

Goal: identify where dynamic dimensions remain reachable from token inputs.

Implementation in this repository:

- `scripts/inspect_onnx_fixed_shape.py`
- `src/fastkokoro/fixed_shape_analysis.py`

Run it against a model:

```bash
uv run python scripts/inspect_onnx_fixed_shape.py /path/to/model.onnx
```

What to look for:

- fixed input tensors immediately feeding dynamic outputs
- `Slice`, `Reshape`, `Shape`, `Gather`, `Expand`, `Pad`, and related ops on the
  token path
- dynamic outputs that remain reachable from `tokens` or `input_ids`

Acceptance criteria:

- the report clearly identifies the first barriers that reintroduce dynamic
  shape
- the report shows whether dynamic tensors remain on the path to the graph
  outputs

### Phase 1: Remove The Input-Only Illusion

Goal: stop treating `fixed input + dynamic slice` as sufficient.

Required change:

- avoid reintroducing the real token length as the dominant shape for the rest
  of the graph

Possible strategies:

- bucket the effective token sequence length across the graph and apply masking
  semantics rather than slicing to exact length
- replace shape-producing subgraphs so downstream consumers see the bucket shape
  instead of the real token length

Acceptance criteria:

- the fixed-shape inspection report no longer shows an early dynamic barrier on
  the token path

### Phase 2: Stabilize Output Buckets

Goal: keep the output path inside bounded, bucketed audio shapes.

Required change:

- use padded output tensors with explicit valid-length handling
- avoid making streamed segments force a fresh output shape every run

Tradeoff:

- some extra compute and copy overhead from padding
- in exchange for lower ORT scheduling/allocation churn

Acceptance criteria:

- output tensors on the first-chunk path stay inside a small set of bucketed
  shapes
- steady-state ORT profiling stops showing high churn from shape-sensitive nodes

### Phase 3: Make Streaming Segmentation Compatible With Buckets

Goal: reduce the number of distinct shapes reaching the model during streaming.

Required change:

- segment scheduling must respect bucket boundaries
- the first streamed segment must fit at least one word while still mapping to a
  stable bucket

Constraints:

- the scheduler should not become O(n²) in phonemization/token counting
- per-segment length selection must be local and predictable

Acceptance criteria:

- TTFC variance falls across mixed prompts
- the first streamed segment does not create a unique shape for every request

### Phase 4: Re-evaluate IOBinding And CUDA Graph

Goal: test the optimizations that depend on stable shapes.

Only after the earlier phases are true:

- compare `onnx_io_binding=true` vs `false`
- evaluate CUDA Graph feasibility

Acceptance criteria:

- TTFC improves on real mixed-prompt benchmarks
- ORT profiling shows less launch/allocation churn
- `IOBinding` becomes clearly beneficial or clearly unnecessary

## Why This Order

Warmup and request replay cannot solve a graph that reconstructs dynamic shapes
on every real request. The expensive path has to become shape-stable before
warmup can be expected to absorb it.

This plan starts by making the shape problem observable, then removes the
specific barriers that keep the graph dynamic, and only then revisits runtime
optimizations such as `IOBinding` and CUDA Graph.
