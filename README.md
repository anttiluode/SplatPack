# SplatPack

**Shared field + sparse residuals for 3D Gaussian splats.**

SplatPack asks a simple question about a trained Gaussian-splat scene:

> How many splat attributes really need to be stored independently?

A normal 3DGS PLY stores RGB / spherical harmonics, opacity, scale and rotation for every splat. SplatPack fits a compact coordinate-conditioned field to those attributes, then stores exact residual attribute vectors only for the splats the field explains worst.

```text
3DGS PLY
   |
   +--> XYZ positions ------------------------------+
   |                                                |
   +--> shared coordinate field F(x,y,z)            |
   |                                                +--> .spk
   +--> top-K hard splats --> exact residuals -------+
```

This is **not** claiming a new state-of-the-art Gaussian-splat codec. Version 0.1 is an inspectable experiment/tool for measuring the **shared-field + residual decomposition** of a real splat scene. The important output is the rate–distortion curve and the identity of the hard residual splats.

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

## What gets modeled

For a standard 3DGS PLY, SplatPack models floating-point properties named like:

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

For each splat, SplatPack measures prediction error after normalizing every modeled property by its data standard deviation:

```math
score_i = mean_j ((a_ij - F_j(q_i)) / sigma_j)^2
```

The worst-scoring fraction becomes the residual set. For those splats, SplatPack stores the full attribute residual vector.

This makes the residual set useful in its own right: it is a map of **where the scene refuses the shared model**. On real captures those regions may concentrate around thin geometry, foliage, occlusion boundaries, specular material, text, or simply failure modes of the chosen field.

## Rate–distortion output

The sweep reports, for every residual fraction:

- actual `.spk` byte size,
- compression ratio against the source PLY,
- number of exact residual splats,
- normalized attribute RMSE.

The metric is currently **attribute-space**, not rendered PSNR/SSIM. A renderer-aware benchmark is the obvious next layer, but keeping v0.1 renderer-independent makes it usable with PLYs from different 3DGS pipelines.

## Useful knobs

```bash
splatpack sweep scene.ply sweep_out \
  --rff 24 \
  --fit-samples 100000 \
  --ridge 1e-3 \
  --fractions 0 0.01 0.02 0.05 0.10 0.20
```

For very large scenes, `--fit-samples` bounds fitting memory; all splats are still scored for residual selection.

## Why this exists

This repo follows experiments in SplatWorld4 where a compact world-space field could generate many splat attributes, but a learned operator substrate lost clean capacity-matched controls. The useful idea survived the failed mechanism claim:

```text
not every splat attribute needs to be independent.
```

SplatPack therefore does not privilege the operator. It starts with a generic compact field and asks whether a real scene decomposes into:

```text
cheap shared structure + sparse exceptions.
```

If the rate–distortion curve bends strongly, that is the signal to build the next version: rendered-quality scoring, residual visualization, stronger interchangeable backbones, and coherent global editing of the shared field.
