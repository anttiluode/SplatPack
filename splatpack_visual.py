#!/usr/bin/env python3
"""Visual diagnostics for SplatPack shared-field residuals."""
from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

import splatpack as core

SH_C0 = 0.28209479177387814


def _numeric_property_names(table: core.PlyTable) -> list[str]:
    return [p.name for p in table.properties if p.name not in {"x", "y", "z"}]


def _dead_properties(table: core.PlyTable) -> list[str]:
    dead = []
    for name in _numeric_property_names(table):
        a = np.asarray(table.data[name])
        if len(a) == 0 or float(np.ptp(a)) == 0.0:
            dead.append(name)
    return dead


def _is_standard_3dgs(table: core.PlyTable) -> bool:
    names = set(table.names)
    required = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    return required.issubset(names)


def _choose_targets(table: core.PlyTable, requested: Sequence[str] | None, mode: str = "auto") -> tuple[list[str], str]:
    names = set(table.names)
    native_3dgs = _is_standard_3dgs(table)

    if requested:
        missing = [n for n in requested if n not in names]
        if missing:
            raise ValueError(f"properties not found: {', '.join(missing)}")
        targets = list(requested)
        chosen_mode = "custom"
    elif {"red", "green", "blue"}.issubset(names):
        targets = ["red", "green", "blue"]
        chosen_mode = "rgb"
    elif native_3dgs and mode in {"auto", "dc"}:
        targets = ["f_dc_0", "f_dc_1", "f_dc_2"]
        chosen_mode = "dc"
    elif native_3dgs and mode == "geometry":
        targets = ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        chosen_mode = "geometry"
    elif native_3dgs and mode == "sh":
        targets = [n for n in table.names if n.startswith("f_dc_") or n.startswith("f_rest_")]
        chosen_mode = "sh"
    else:
        targets = core.gaussian_modeled_properties(table)
        chosen_mode = "all"

    live = []
    for name in targets:
        a = np.asarray(table.data[name])
        if len(a) and float(np.ptp(a)) > 0.0:
            live.append(name)
    if not live:
        raise ValueError("selected properties are constant; nothing nontrivial to visualize")
    return live, chosen_mode


def _heat_colors(scores: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.sqrt(np.maximum(np.asarray(scores, dtype=np.float64), 0.0))
    positive = d[d > 0]
    scale = float(np.percentile(positive, 95.0)) if len(positive) else 1.0
    scale = max(scale, 1e-12)
    t = np.clip(d / scale, 0.0, 1.0)
    anchors_t = np.array([0.0, 0.33, 0.66, 1.0])
    anchors_rgb = np.array([[25, 55, 170], [0, 205, 235], [255, 225, 25], [220, 20, 20]], dtype=np.float64)
    rgb = np.empty((len(t), 3), dtype=np.uint8)
    for i, u in enumerate(t):
        j = min(len(anchors_t) - 2, int(np.searchsorted(anchors_t, u, side="right") - 1))
        j = max(0, j)
        a, b = anchors_t[j], anchors_t[j + 1]
        w = 0.0 if b == a else (u - a) / (b - a)
        rgb[i] = np.clip(np.rint((1.0 - w) * anchors_rgb[j] + w * anchors_rgb[j + 1]), 0, 255)
    return rgb, scale


def _xyz(table: core.PlyTable, idx=None) -> dict[str, np.ndarray]:
    if idx is None:
        idx = slice(None)
    return {n: np.asarray(table.data[n][idx]) for n in ("x", "y", "z")}


def _point_props(include_score: bool = False) -> list[core.Property]:
    props = [
        core.Property("x", "float"), core.Property("y", "float"), core.Property("z", "float"),
        core.Property("red", "uchar"), core.Property("green", "uchar"), core.Property("blue", "uchar"),
    ]
    if include_score:
        props.append(core.Property("residual", "float"))
    return props


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    rgb01 = np.asarray(rgb, dtype=np.float32) / 255.0
    return ((rgb01 - 0.5) / SH_C0).astype(np.float32)


def _write_3dgs_ply(path: str | Path, table: core.PlyTable, *, selected_idx: np.ndarray | None = None,
                     display_rgb: np.ndarray | None = None, dc_values: np.ndarray | None = None,
                     zero_rest: bool = False, comment: str = "SplatPack 3DGS diagnostic",
                     chunk: int = 100_000) -> None:
    """Write viewer-compatible 3DGS PLY while preserving Gaussian geometry."""
    if not _is_standard_3dgs(table):
        raise ValueError("Gaussian-native export requires standard 3DGS properties")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if selected_idx is None:
        count = table.count
    else:
        selected_idx = np.asarray(selected_idx, dtype=np.int64)
        count = len(selected_idx)
    if display_rgb is not None:
        display_rgb = np.asarray(display_rgb)
        expected = table.count if selected_idx is None else count
        if len(display_rgb) != expected:
            raise ValueError("display_rgb length does not match output vertex count")
    if dc_values is not None:
        dc_values = np.asarray(dc_values, dtype=np.float32)
        expected = table.count if selected_idx is None else count
        if dc_values.shape != (expected, 3):
            raise ValueError("dc_values must have shape (output_count, 3)")

    lines = ["ply", "format binary_little_endian 1.0"]
    for c in table.comments:
        lines.append(f"comment {c}")
    lines.append(f"comment {comment}")
    for o in table.obj_info:
        lines.append(f"obj_info {o}")
    lines.append(f"element vertex {count}")
    for p in table.properties:
        lines.append(f"property {p.ply_type} {p.name}")
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")
    rest_names = [n for n in table.names if n.startswith("f_rest_")]
    source_dtype = table.data.dtype

    with path.open("wb") as f:
        f.write(header)
        for start in range(0, count, chunk):
            stop = min(count, start + chunk)
            src_idx = slice(start, stop) if selected_idx is None else selected_idx[start:stop]
            arr = np.empty(stop - start, dtype=source_dtype)
            for p in table.properties:
                arr[p.name] = np.asarray(table.data[p.name][src_idx])
            if display_rgb is not None:
                sh = _rgb_to_sh_dc(display_rgb[start:stop])
                arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh[:, 0], sh[:, 1], sh[:, 2]
                for name in rest_names:
                    arr[name] = 0
            elif dc_values is not None:
                dc = dc_values[start:stop]
                arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
                if zero_rest:
                    for name in rest_names:
                        arr[name] = 0
            elif zero_rest:
                for name in rest_names:
                    arr[name] = 0
            arr.tofile(f)


def _write_point_cloud_outputs(table: core.PlyTable, out: Path, stem: str, targets: Sequence[str],
                               colors: np.ndarray, scores: np.ndarray, hard_idx: np.ndarray,
                               hard_fraction: float, fit: core.FieldFit) -> tuple[Path, Path, Path | None]:
    heat_path = out / f"{stem}_residual_heatmap.ply"
    heat_cols = _xyz(table)
    heat_cols.update({"red": colors[:, 0], "green": colors[:, 1], "blue": colors[:, 2],
                      "residual": np.sqrt(np.maximum(scores, 0)).astype(np.float32)})
    core.write_ply(heat_path, _point_props(True), heat_cols,
                   comments=[f"SplatPack residual heatmap targets={','.join(targets)}"], fmt="binary_little_endian")

    hard_path = out / f"{stem}_hard_{int(round(hard_fraction * 100)):02d}pct.ply"
    hard_cols = _xyz(table, hard_idx)
    if {"red", "green", "blue"}.issubset(table.names):
        hard_rgb = np.column_stack([np.asarray(table.data["red"][hard_idx]),
                                    np.asarray(table.data["green"][hard_idx]),
                                    np.asarray(table.data["blue"][hard_idx])]).astype(np.uint8)
    else:
        hard_rgb = colors[hard_idx]
    hard_cols.update({"red": hard_rgb[:, 0], "green": hard_rgb[:, 1], "blue": hard_rgb[:, 2],
                      "residual": np.sqrt(np.maximum(scores[hard_idx], 0)).astype(np.float32)})
    core.write_ply(hard_path, _point_props(True), hard_cols,
                   comments=[f"SplatPack hardest {hard_fraction:.1%} targets={','.join(targets)}"], fmt="binary_little_endian")

    shared_path = None
    if list(targets) == ["red", "green", "blue"]:
        pred = core.predict_field(fit, core.coords_from_table(table))
        pred = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
        shared_path = out / f"{stem}_shared_rgb.ply"
        shared_cols = _xyz(table)
        shared_cols.update({"red": pred[:, 0], "green": pred[:, 1], "blue": pred[:, 2]})
        core.write_ply(shared_path, _point_props(False), shared_cols,
                       comments=["SplatPack field-only RGB prediction"], fmt="binary_little_endian")
    return heat_path, hard_path, shared_path


def _write_3dgs_outputs(table: core.PlyTable, out: Path, stem: str, targets: Sequence[str],
                        colors: np.ndarray, hard_idx: np.ndarray, hard_fraction: float,
                        fit: core.FieldFit, chunk: int) -> tuple[Path, Path, Path | None]:
    heat_path = out / f"{stem}_residual_heatmap.ply"
    _write_3dgs_ply(heat_path, table, display_rgb=colors,
                     comment=f"SplatPack Gaussian-native residual heatmap targets={','.join(targets)}", chunk=chunk)
    hard_path = out / f"{stem}_hard_{int(round(hard_fraction * 100)):02d}pct.ply"
    _write_3dgs_ply(hard_path, table, selected_idx=hard_idx,
                     comment=f"SplatPack hardest {hard_fraction:.1%} Gaussians; original appearance preserved", chunk=chunk)
    shared_path = None
    if list(targets) == ["f_dc_0", "f_dc_1", "f_dc_2"]:
        pred_dc = core.predict_field(fit, core.coords_from_table(table))
        shared_path = out / f"{stem}_shared_dc.ply"
        _write_3dgs_ply(shared_path, table, dc_values=pred_dc, zero_rest=True,
                         comment="SplatPack field-only SH DC prediction; higher-order SH zeroed", chunk=chunk)
    return heat_path, hard_path, shared_path


def run_visual(input_path: str | Path, out_dir: str | Path, properties: Sequence[str] | None = None,
               hard_fraction: float = 0.10, n_rff: int = 24, ridge: float = 1e-3,
               fit_samples: int = 100_000, seed: int = 0, chunk: int = 100_000,
               mode: str = "auto") -> dict:
    table = core.read_ply(input_path)
    targets, chosen_mode = _choose_targets(table, properties, mode)
    fit = core.fit_field(table, targets, n_rff=n_rff, ridge=ridge, fit_samples=fit_samples, seed=seed)
    scores, report = core.residual_scores(table, fit, chunk=chunk)
    colors, heat_scale = _heat_colors(scores)
    hard_idx = core.choose_residual_indices(scores, hard_fraction)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    native_3dgs = _is_standard_3dgs(table)

    if native_3dgs:
        heat_path, hard_path, shared_path = _write_3dgs_outputs(
            table, out, stem, targets, colors, hard_idx, hard_fraction, fit, chunk)
        output_mode = "gaussian-native"
    else:
        heat_path, hard_path, shared_path = _write_point_cloud_outputs(
            table, out, stem, targets, colors, scores, hard_idx, hard_fraction, fit)
        output_mode = "point-cloud"

    summary = {
        "input": str(input_path), "splats": table.count, "targets": targets,
        "target_mode": chosen_mode, "output_mode": output_mode,
        "dead_properties": _dead_properties(table), "feature_count": fit.feature_count,
        "field_attribute_nrmse": report["attribute_nrmse"],
        "per_attribute_nrmse": report["per_attribute_nrmse"],
        "hard_fraction": float(hard_fraction), "hard_points": int(len(hard_idx)),
        "heat_scale_p95_normalized_residual": heat_scale,
        "residual_heatmap": str(heat_path), "hard_points_file": str(hard_path),
        "shared_field": str(shared_path) if shared_path else None,
        "shared_rgb": str(shared_path) if shared_path and chosen_mode == "rgb" else None,
        "shared_dc": str(shared_path) if shared_path and chosen_mode == "dc" else None,
    }
    if native_3dgs:
        summary["viewer_note"] = (
            "Outputs preserve XYZ, opacity, scale and rotation. The residual heatmap replaces SH color only "
            "and zeros higher-order SH so Gaussian viewers render diagnostic colors directly."
        )
    summary_path = out / f"{stem}_visual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def _make_3dgs_fixture(path: Path, n: int = 300, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    x, y, z = xyz.T
    fdc = np.column_stack([0.8 * np.sin(x), 0.7 * np.cos(y), 0.6 * np.sin(z + x)]).astype(np.float32)
    frest = rng.normal(0, 0.08, size=(n, 6)).astype(np.float32)
    opacity = rng.normal(2.0, 0.4, size=n).astype(np.float32)
    scale = rng.normal(-3.0, 0.2, size=(n, 3)).astype(np.float32)
    rot = np.zeros((n, 4), np.float32); rot[:, 0] = 1.0; rot[:, 1:] = rng.normal(0, 0.03, size=(n, 3))
    zeros = np.zeros(n, np.float32)
    props = [core.Property("x", "float"), core.Property("y", "float"), core.Property("z", "float"),
             core.Property("nx", "float"), core.Property("ny", "float"), core.Property("nz", "float"),
             core.Property("f_dc_0", "float"), core.Property("f_dc_1", "float"), core.Property("f_dc_2", "float")]
    props += [core.Property(f"f_rest_{i}", "float") for i in range(6)]
    props += [core.Property("opacity", "float"), core.Property("scale_0", "float"), core.Property("scale_1", "float"),
              core.Property("scale_2", "float"), core.Property("rot_0", "float"), core.Property("rot_1", "float"),
              core.Property("rot_2", "float"), core.Property("rot_3", "float")]
    cols = {"x": x, "y": y, "z": z, "nx": zeros, "ny": zeros, "nz": zeros,
            "f_dc_0": fdc[:, 0], "f_dc_1": fdc[:, 1], "f_dc_2": fdc[:, 2], "opacity": opacity,
            "scale_0": scale[:, 0], "scale_1": scale[:, 1], "scale_2": scale[:, 2],
            "rot_0": rot[:, 0], "rot_1": rot[:, 1], "rot_2": rot[:, 2], "rot_3": rot[:, 3]}
    for i in range(6):
        cols[f"f_rest_{i}"] = frest[:, i]
    core.write_ply(path, props, cols, fmt="binary_little_endian")


def _selftest() -> None:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        n = 400
        xyz = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
        x, y, z = xyz.T
        rgb = np.column_stack([128 + 90 * np.sin(2 * x), 110 + 80 * np.cos(2 * y), 100 + 75 * np.sin(z + x)])
        hard = (x > 0.55) & (y > 0.35)
        rgb[hard] += rng.normal(0, 70, size=(hard.sum(), 3))
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        props = [core.Property("x", "float"), core.Property("y", "float"), core.Property("z", "float"),
                 core.Property("nx", "float"), core.Property("ny", "float"), core.Property("nz", "float"),
                 core.Property("red", "uchar"), core.Property("green", "uchar"), core.Property("blue", "uchar")]
        zeros = np.zeros(n, np.float32)
        cols = {"x": x, "y": y, "z": z, "nx": zeros, "ny": zeros, "nz": zeros,
                "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2]}
        src = td / "visual_test.ply"
        core.write_ply(src, props, cols, fmt="ascii")
        summary = run_visual(src, td / "out", hard_fraction=0.10, n_rff=8, fit_samples=350, seed=7)
        assert summary["targets"] == ["red", "green", "blue"] and summary["output_mode"] == "point-cloud"
        assert set(summary["dead_properties"]) >= {"nx", "ny", "nz"}
        assert Path(summary["residual_heatmap"]).exists() and Path(summary["hard_points_file"]).exists()
        assert Path(summary["shared_rgb"]).exists()

        gs_src = td / "gaussians.ply"
        _make_3dgs_fixture(gs_src)
        gs = run_visual(gs_src, td / "gs_out", hard_fraction=0.10, n_rff=8, fit_samples=250, seed=11)
        assert gs["targets"] == ["f_dc_0", "f_dc_1", "f_dc_2"] and gs["target_mode"] == "dc"
        assert gs["output_mode"] == "gaussian-native" and Path(gs["shared_dc"]).exists()
        original = core.read_ply(gs_src); heat = core.read_ply(gs["residual_heatmap"]); shared = core.read_ply(gs["shared_dc"])
        assert heat.names == original.names and shared.names == original.names
        for name in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
            assert np.allclose(np.asarray(heat.data[name]), np.asarray(original.data[name]))
        for name in [n for n in heat.names if n.startswith("f_rest_")]:
            assert np.all(np.asarray(heat.data[name]) == 0) and np.all(np.asarray(shared.data[name]) == 0)
        del summary, gs, original, heat, shared
        gc.collect()
    print("SplatPack visual selftest: PASS")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="splatpack-heatmap",
        description="Visualize shared-field failures; standard 3DGS inputs stay as viewer-compatible Gaussians.")
    ap.add_argument("input", help="PLY input, or 'selftest'")
    ap.add_argument("out_dir", nargs="?", default="splatpack_visual")
    ap.add_argument("--properties", nargs="+", default=None, help="properties to model; overrides --mode")
    ap.add_argument("--mode", choices=["auto", "dc", "geometry", "sh", "all"], default="auto",
                    help="standard 3DGS target family; auto defaults to f_dc_0..2")
    ap.add_argument("--hard", type=float, default=0.10, help="fraction of hardest points to export")
    ap.add_argument("--rff", type=int, default=24)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fit-samples", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=100_000)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.input == "selftest":
        _selftest(); return
    try:
        result = run_visual(args.input, args.out_dir, args.properties, args.hard, args.rff,
                            args.ridge, args.fit_samples, args.seed, args.chunk, args.mode)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        raise SystemExit(f"SplatPack visual error: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
