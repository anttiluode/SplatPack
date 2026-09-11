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


def _numeric_property_names(table: core.PlyTable) -> list[str]:
    return [p.name for p in table.properties if p.name not in {"x", "y", "z"}]


def _dead_properties(table: core.PlyTable) -> list[str]:
    dead = []
    for name in _numeric_property_names(table):
        a = np.asarray(table.data[name], dtype=np.float64)
        if len(a) == 0 or float(np.ptp(a)) == 0.0:
            dead.append(name)
    return dead


def _choose_targets(table: core.PlyTable, requested: Sequence[str] | None) -> list[str]:
    names = set(table.names)
    if requested:
        missing = [n for n in requested if n not in names]
        if missing:
            raise ValueError(f"properties not found: {', '.join(missing)}")
        targets = list(requested)
    elif {"red", "green", "blue"}.issubset(names):
        targets = ["red", "green", "blue"]
    else:
        targets = core.gaussian_modeled_properties(table)
    live = []
    for name in targets:
        a = np.asarray(table.data[name], dtype=np.float64)
        if len(a) and float(np.ptp(a)) > 0.0:
            live.append(name)
    if not live:
        raise ValueError("selected properties are constant; nothing nontrivial to visualize")
    return live


def _heat_colors(scores: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.sqrt(np.maximum(np.asarray(scores, dtype=np.float64), 0.0))
    positive = d[d > 0]
    scale = float(np.percentile(positive, 95.0)) if len(positive) else 1.0
    scale = max(scale, 1e-12)
    t = np.clip(d / scale, 0.0, 1.0)
    anchors_t = np.array([0.0, 0.33, 0.66, 1.0])
    anchors_rgb = np.array([
        [25, 55, 170],
        [0, 205, 235],
        [255, 225, 25],
        [220, 20, 20],
    ], dtype=np.float64)
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


def run_visual(input_path: str | Path, out_dir: str | Path, properties: Sequence[str] | None = None,
               hard_fraction: float = 0.10, n_rff: int = 24, ridge: float = 1e-3,
               fit_samples: int = 100_000, seed: int = 0, chunk: int = 100_000) -> dict:
    table = core.read_ply(input_path)
    targets = _choose_targets(table, properties)
    fit = core.fit_field(table, targets, n_rff=n_rff, ridge=ridge, fit_samples=fit_samples, seed=seed)
    scores, report = core.residual_scores(table, fit, chunk=chunk)
    colors, heat_scale = _heat_colors(scores)
    hard_idx = core.choose_residual_indices(scores, hard_fraction)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    heat_path = out / f"{stem}_residual_heatmap.ply"
    heat_cols = _xyz(table)
    heat_cols.update({
        "red": colors[:, 0], "green": colors[:, 1], "blue": colors[:, 2],
        "residual": np.sqrt(np.maximum(scores, 0)).astype(np.float32),
    })
    core.write_ply(heat_path, _point_props(include_score=True), heat_cols,
                   comments=[f"SplatPack residual heatmap targets={','.join(targets)}"], fmt="binary_little_endian")

    hard_path = out / f"{stem}_hard_{int(round(hard_fraction * 100)):02d}pct.ply"
    hard_cols = _xyz(table, hard_idx)
    if {"red", "green", "blue"}.issubset(table.names):
        hard_rgb = np.column_stack([
            np.asarray(table.data["red"][hard_idx]),
            np.asarray(table.data["green"][hard_idx]),
            np.asarray(table.data["blue"][hard_idx]),
        ]).astype(np.uint8)
    else:
        hard_rgb = colors[hard_idx]
    hard_cols.update({
        "red": hard_rgb[:, 0], "green": hard_rgb[:, 1], "blue": hard_rgb[:, 2],
        "residual": np.sqrt(np.maximum(scores[hard_idx], 0)).astype(np.float32),
    })
    core.write_ply(hard_path, _point_props(include_score=True), hard_cols,
                   comments=[f"SplatPack hardest {hard_fraction:.1%} targets={','.join(targets)}"], fmt="binary_little_endian")

    shared_path = None
    if targets == ["red", "green", "blue"]:
        pred = core.predict_field(fit, core.coords_from_table(table))
        pred = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
        shared_path = out / f"{stem}_shared_rgb.ply"
        shared_cols = _xyz(table)
        shared_cols.update({"red": pred[:, 0], "green": pred[:, 1], "blue": pred[:, 2]})
        core.write_ply(shared_path, _point_props(include_score=False), shared_cols,
                       comments=["SplatPack field-only RGB prediction"], fmt="binary_little_endian")

    summary = {
        "input": str(input_path),
        "splats": table.count,
        "targets": targets,
        "dead_properties": _dead_properties(table),
        "feature_count": fit.feature_count,
        "field_attribute_nrmse": report["attribute_nrmse"],
        "per_attribute_nrmse": report["per_attribute_nrmse"],
        "hard_fraction": float(hard_fraction),
        "hard_points": int(len(hard_idx)),
        "heat_scale_p95_normalized_residual": heat_scale,
        "residual_heatmap": str(heat_path),
        "hard_points_file": str(hard_path),
        "shared_rgb": str(shared_path) if shared_path else None,
    }
    summary_path = out / f"{stem}_visual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def _selftest() -> None:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        n = 400
        xyz = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
        x, y, z = xyz.T
        rgb = np.column_stack([
            128 + 90 * np.sin(2 * x),
            110 + 80 * np.cos(2 * y),
            100 + 75 * np.sin(z + x),
        ])
        hard = (x > 0.55) & (y > 0.35)
        rgb[hard] += rng.normal(0, 70, size=(hard.sum(), 3))
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        props = [
            core.Property("x", "float"), core.Property("y", "float"), core.Property("z", "float"),
            core.Property("nx", "float"), core.Property("ny", "float"), core.Property("nz", "float"),
            core.Property("red", "uchar"), core.Property("green", "uchar"), core.Property("blue", "uchar"),
        ]
        zeros = np.zeros(n, np.float32)
        cols = {"x": x, "y": y, "z": z, "nx": zeros, "ny": zeros, "nz": zeros,
                "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2]}
        src = td / "visual_test.ply"
        core.write_ply(src, props, cols, fmt="ascii")
        summary = run_visual(src, td / "out", hard_fraction=0.10, n_rff=8, fit_samples=350, seed=7)
        assert summary["targets"] == ["red", "green", "blue"]
        assert set(summary["dead_properties"]) >= {"nx", "ny", "nz"}
        assert Path(summary["residual_heatmap"]).exists()
        assert Path(summary["hard_points_file"]).exists()
        assert Path(summary["shared_rgb"]).exists()
        del summary
        gc.collect()
    print("SplatPack visual selftest: PASS")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="splatpack-heatmap",
        description="Visualize where a compact XYZ-conditioned field fails to explain point attributes.",
    )
    ap.add_argument("input", help="PLY input, or 'selftest'")
    ap.add_argument("out_dir", nargs="?", default="splatpack_visual")
    ap.add_argument("--properties", nargs="+", default=None,
                    help="properties to model; default prefers red green blue when present")
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
        _selftest()
        return
    try:
        result = run_visual(args.input, args.out_dir, args.properties, args.hard, args.rff,
                            args.ridge, args.fit_samples, args.seed, args.chunk)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        raise SystemExit(f"SplatPack visual error: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
