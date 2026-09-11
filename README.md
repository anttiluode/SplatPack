# SplatPack

**Shared field + sparse residuals for 3D Gaussian splats and point-cloud attributes.**

SplatPack asks a simple question about a scene:

> How many per-point attributes really need to be stored independently?

A normal 3DGS PLY stores RGB / spherical harmonics, opacity, scale and rotation for every splat. SplatPack fits a compact coordinate-conditioned field to those attributes, then stores exact residual attribute vectors only for the splats the field explains worst.

```text
3DGS / point-cloud PLY
   |
   +--> XYZ positions ------------------------------+
   |                                                |
   +--> shared coordinate field F(x,y,z)            |
   |                                                +--> .spk
   +--> top-K hard points --> exact residuals -------+
```

This is **not** claiming a new state-of-the-art Gaussian-splat codec. Version 0.1 is an inspectable experiment/tool for measuring the **shared-field + residual decomposition** of a real scene. The important outputs are the rate–distortion curve and the identity of the hard residual points.

## Install

```bash
python -m pip install -e .
```

Only NumPy is required.

## Quick start

Inspect a PLY:

```bash
splatpack analyze point_cloud.ply
```

Build the default rate–distortion sweep:

```bash
splatpack sweep point_cloud.ply sweep_out
```

The sweep fits the shared field once and writes packs at residual fractions:

```text
0%, 1%, 2%, 5%, 10%, 20%
```

It produces:

```text
sweep_out/
  point_cloud.r0p0000.spk
  point_cloud.r0p0100.spk
  ...
  splatpack_sweep.json
  splatpack_sweep.csv
```

Compress at one operating point:

```bash
splatpack compress point_cloud.ply scene.spk --residual 0.05
```

Reconstruct a standard PLY:

```bash
splatpack decompress scene.spk restored.ply
```

Run a synthetic demo with a deliberately difficult local region:

```bash
splatpack demo demo_out
```

Run the built-in round-trip check:

```bash
splatpack selftest
```

## Visual residual diagnostics

Some point-cloud PLY exporters include RGB plus placeholder normal channels such as `nx=ny=nz=0`. A zero reconstruction error on those channels is trivial, not evidence that the shape was learned.

For visual inspection, use:

```bash
splatpack-heatmap input.ply visual_out
```

If `red`, `green`, and `blue` are present, the visual tool models RGB by default. Otherwise it falls back to the available non-constant modeled attributes. It also reports constant/dead properties.

The command writes:

```text
visual_out/
  input_residual_heatmap.ply   # blue/cyan = easy, yellow/red = hard
  input_hard_10pct.ply         # only the hardest 10% of points
  input_shared_rgb.ply         # field-only RGB prediction, when RGB exists
  input_visual_summary.json
```

The heatmap answers a different question from ordinary compression:

> Where does a small shared coordinate field stop explaining the object?

That can expose boundaries, local detail, texture changes, thin geometry, specular regions, or simply weaknesses of the chosen field.

Choose other attributes explicitly when useful:

```bash
splatpack-heatmap point_cloud.ply visual_out --properties opacity scale_0 scale_1 scale_2
```

Change how much of the hard set is exported:

```bash
splatpack-heatmap input.ply visual_out --hard 0.05
```

## What gets modeled

For a standard 3DGS PLY, the compressor models floating-point properties named like:

```text
f_dc_*
f_rest_*
opacity
scale_*
rot_*
```

`x`, `y`, `z` are stored directly. Other properties are passed through exactly. Version 0.1 supports vertex-only ASCII and binary little-endian PLY files with scalar vertex properties.

The shared field is deliberately simple and CPU-friendly:

```text
normalized XYZ
  -> quadratic coordinate terms
  -> deterministic random Fourier features
  -> ridge-regression attribute head
```

The default field has 58 basis features (`10 + 2*24`) regardless of the number of splats.

## Residual selection

For each point, SplatPack measures prediction error after normalizing every modeled property by its data standard deviation:

```math
score_i = mean_j ((a_ij - F_j(q_i)) / sigma_j)^2
```

The worst-scoring fraction becomes the residual set. For those points, SplatPack stores the full attribute residual vector.

This makes the residual set useful in its own right: it is a map of **where the scene refuses the shared model**. On real captures those regions may concentrate around thin geometry, foliage, occlusion boundaries, specular material, text, or simply failure modes of the chosen field.

## Rate–distortion output

The sweep reports, for every residual fraction:

- actual `.spk` byte size,
- compression ratio against the source PLY,
- number of exact residual points,
- normalized attribute RMSE.

The metric is currently **attribute-space**, not rendered PSNR/SSIM. A renderer-aware benchmark is the obvious next layer, but keeping v0.1 renderer-independent makes it usable with PLYs from different pipelines.

## Useful knobs

```bash
splatpack sweep scene.ply sweep_out \
  --rff 24 \
  --fit-samples 100000 \
  --ridge 1e-3 \
  --fractions 0 0.01 0.02 0.05 0.10 0.20
```

For very large scenes, `--fit-samples` bounds fitting memory; all points are still scored for residual selection.

## Why this exists

This repo follows experiments in SplatWorld4 where a compact world-space field could generate many splat attributes, but a learned operator substrate lost clean capacity-matched controls. The useful idea survived the failed mechanism claim:

```text
not every splat attribute needs to be independent.
```

SplatPack therefore does not privilege the operator. It starts with a generic compact field and asks whether a real scene decomposes into:

```text
cheap shared structure + sparse exceptions.
```

If the rate–distortion curve bends strongly, that is the signal to build the next version: rendered-quality scoring, stronger interchangeable backbones, and coherent global editing of the shared field.
