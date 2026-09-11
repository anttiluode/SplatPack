# SplatPack

**Shared field + sparse residuals for 3D Gaussian splats and point-cloud attributes.**

SplatPack asks a simple question about a scene:

> How many per-point attributes really need to be stored independently?

A normal 3DGS PLY stores spherical harmonics, opacity, scale and rotation for every splat. SplatPack fits a compact coordinate-conditioned field to those attributes, then stores exact residual attribute vectors only for the splats the field explains worst.

```text
3DGS / point-cloud PLY
   |
   +--> XYZ positions ------------------------------+
   |                                                |
   +--> shared coordinate field F(x,y,z)            |
   |                                                +--> .spk
   +--> top-K hard points --> exact residuals -------+
```

This is **not** claiming a new state-of-the-art Gaussian-splat codec. It is an inspectable experiment/tool for measuring the **shared-field + residual decomposition** of a real scene. The important outputs are the rate–distortion curve and the identity of the hard residual points.

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

For visual inspection, use:

```bash
splatpack-heatmap point_cloud.ply visual_out
```

### Standard 3DGS input

A standard Gaussian-splat PLY is **not** an ordinary XYZ+RGB point cloud. The visualizer therefore preserves the complete Gaussian record: XYZ, opacity, scale and rotation stay unchanged.

By default the visual question is deliberately narrow and interpretable:

```text
XYZ -> f_dc_0, f_dc_1, f_dc_2
```

These are the three DC spherical-harmonic appearance coefficients. The command writes:

```text
visual_out/
  point_cloud_residual_heatmap.ply  # same Gaussians, blue/cyan easy -> yellow/red hard
  point_cloud_hard_10pct.ply        # hardest 10% of original Gaussians, original appearance
  point_cloud_shared_dc.ply         # field-only DC prediction on original Gaussian geometry
  point_cloud_visual_summary.json
```

`*_residual_heatmap.ply` remains a real Gaussian-splat PLY. Only its SH color is replaced for visualization; higher-order SH terms are zeroed so the diagnostic color is view-independent. Open it in a Gaussian-splat viewer, not a generic polygon-mesh viewer.

`*_shared_dc.ply` answers a particularly useful question:

> What appearance does the tiny shared XYZ-conditioned field predict if the individual color coefficients are removed?

Other target families can be inspected separately:

```bash
splatpack-heatmap point_cloud.ply visual_out --mode geometry
splatpack-heatmap point_cloud.ply visual_out --mode sh
splatpack-heatmap point_cloud.ply visual_out --mode all
```

Or choose exact attributes:

```bash
splatpack-heatmap point_cloud.ply visual_out --properties opacity scale_0 scale_1 scale_2
```

Keeping these questions separate matters: a single score over color, high-order SH, opacity, scale and rotation can be hard to interpret.

### Ordinary point-cloud input

If `red`, `green`, and `blue` are present, the visualizer models RGB by default. Some exporters also include placeholder normal channels such as `nx=ny=nz=0`; those are reported as dead/constant rather than treated as a successful fit.

For non-3DGS point clouds the output is an ordinary XYZ+RGB residual map.

Change how much of the hard set is exported with:

```bash
splatpack-heatmap input.ply visual_out --hard 0.05
```

The heatmap answers a different question from ordinary compression:

> Where does a small shared coordinate field stop explaining the object?

A bad field fit is a valid result. The diagnostic should show its failure rather than turn a high error into a compression claim.

## What gets modeled

For a standard 3DGS PLY, the compressor models floating-point properties named like:

```text
f_dc_*
f_rest_*
opacity
scale_*
rot_*
```

`x`, `y`, `z` are stored directly. Other properties are passed through exactly. The current format supports vertex-only ASCII and binary little-endian PLY files with scalar vertex properties.

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

The metric is currently **attribute-space**, not rendered PSNR/SSIM. A renderer-aware benchmark is the obvious next layer, but keeping the packer renderer-independent makes it usable with PLYs from different pipelines.

## Useful knobs

```bash
splatpack sweep scene.ply sweep_out \
  --rff 24 \
  --fit-samples 100000 \
  --ridge 1e-3 \
  --fractions 0 0.01 0.02 0.05 0.10 0.20
```

For very large scenes, `--fit-samples` bounds fitting memory; all points are still scored for residual selection. Gaussian-native diagnostic files are written in chunks so multi-million-splat scenes do not require another full structured scene in RAM.

## Why this exists

This repo follows experiments in SplatWorld4 where a compact world-space field could generate many splat attributes, but a learned operator substrate lost clean capacity-matched controls. The useful idea survived the failed mechanism claim:

```text
not every splat attribute needs to be independent.
```

SplatPack therefore does not privilege the operator. It starts with a generic compact field and asks whether a real scene decomposes into:

```text
cheap shared structure + sparse exceptions.
```

If the rate–distortion curve bends strongly, that is the signal to try rendered-quality scoring, stronger interchangeable backbones, spatially local fields, and coherent global editing of the shared field.
