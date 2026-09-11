#!/usr/bin/env python3
"""SplatPack: shared-field + sparse-residual compression for Gaussian-splat PLY files.

The first version is intentionally small and inspectable. It learns a compact
coordinate field for floating-point per-splat attributes, keeps XYZ exactly,
and stores full residual attribute vectors only for the splats the field
explains worst.

No renderer is assumed. Quality is therefore measured in normalized attribute
space. A rendered benchmark can be layered on top later without changing the
pack format.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

VERSION = "0.1.0"

PLY_DTYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2",
    "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4",
    "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4",
    "double": "<f8", "float64": "<f8",
}


@dataclass
class Property:
    name: str
    ply_type: str

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(PLY_DTYPES[self.ply_type])

    @property
    def is_float(self) -> bool:
        return self.dtype.kind == "f"


@dataclass
class PlyTable:
    path: Path
    fmt: str
    count: int
    properties: list[Property]
    data: np.ndarray
    comments: list[str]
    obj_info: list[str]

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.properties]


@dataclass
class FieldFit:
    modeled_names: list[str]
    target_mean: np.ndarray
    target_std: np.ndarray
    coord_center: np.ndarray
    coord_scale: np.ndarray
    freq_matrix: np.ndarray
    weights: np.ndarray
    ridge: float
    seed: int

    @property
    def feature_count(self) -> int:
        return int(self.weights.shape[0])


def _read_header(path: Path):
    comments: list[str] = []
    obj_info: list[str] = []
    elements: list[dict] = []
    fmt = None
    offset = 0
    with path.open("rb") as f:
        first = f.readline()
        offset += len(first)
        if first.strip() != b"ply":
            raise ValueError("not a PLY file (missing 'ply' magic)")
        current = None
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("truncated PLY header")
            offset += len(raw)
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            if not line:
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "format":
                if len(parts) < 3:
                    raise ValueError("malformed PLY format line")
                fmt = parts[1]
            elif tag == "comment":
                comments.append(line[len("comment"):].lstrip())
            elif tag == "obj_info":
                obj_info.append(line[len("obj_info"):].lstrip())
            elif tag == "element":
                if len(parts) != 3:
                    raise ValueError("malformed PLY element line")
                current = {"name": parts[1], "count": int(parts[2]), "properties": []}
                elements.append(current)
            elif tag == "property":
                if current is None:
                    raise ValueError("PLY property before element")
                if parts[1] == "list":
                    current["properties"].append(("list", parts[2], parts[3], parts[4]))
                else:
                    current["properties"].append(("scalar", parts[1], parts[2]))
    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"unsupported PLY format {fmt!r}; use ascii or binary_little_endian")
    return fmt, elements, comments, obj_info, offset


def read_ply(path: str | Path) -> PlyTable:
    path = Path(path)
    fmt, elements, comments, obj_info, offset = _read_header(path)
    vertex = next((e for e in elements if e["name"] == "vertex"), None)
    if vertex is None:
        raise ValueError("PLY has no vertex element")
    others = [e for e in elements if e["name"] != "vertex" and e["count"] > 0]
    if others:
        desc = ", ".join(f"{e['name']}={e['count']}" for e in others)
        raise ValueError(f"SplatPack v0.1 supports vertex-only PLY files; extra elements: {desc}")
    props: list[Property] = []
    fields = []
    for item in vertex["properties"]:
        if item[0] != "scalar":
            raise ValueError("SplatPack v0.1 does not support list properties on vertices")
        _, typ, name = item
        if typ not in PLY_DTYPES:
            raise ValueError(f"unsupported PLY scalar type {typ!r}")
        props.append(Property(name=name, ply_type=typ))
        fields.append((name, np.dtype(PLY_DTYPES[typ])))
    dtype = np.dtype(fields, align=False)
    n = int(vertex["count"])
    if fmt == "binary_little_endian":
        needed = offset + n * dtype.itemsize
        if path.stat().st_size < needed:
            raise ValueError("truncated binary PLY vertex data")
        data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(n,))
    else:
        with path.open("rb") as f:
            f.seek(offset)
            data = np.loadtxt(f, dtype=dtype, max_rows=n, ndmin=1)
        if len(data) != n:
            raise ValueError(f"expected {n} vertices, read {len(data)}")
    return PlyTable(path=path, fmt=fmt, count=n, properties=props, data=data,
                    comments=comments, obj_info=obj_info)


def write_ply(path: str | Path, properties: Sequence[Property], columns: dict[str, np.ndarray],
              comments: Sequence[str] = (), obj_info: Sequence[str] = (), fmt: str = "binary_little_endian"):
    path = Path(path)
    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError("fmt must be ascii or binary_little_endian")
    if not properties:
        raise ValueError("no properties")
    n = len(columns[properties[0].name])
    fields = [(p.name, p.dtype) for p in properties]
    arr = np.empty(n, dtype=np.dtype(fields, align=False))
    for p in properties:
        col = np.asarray(columns[p.name])
        if len(col) != n:
            raise ValueError(f"property {p.name} has wrong length")
        arr[p.name] = col.astype(p.dtype, copy=False)
    lines = ["ply", f"format {fmt} 1.0"]
    for c in comments:
        lines.append(f"comment {c}")
    lines.append(f"comment SplatPack {VERSION} reconstructed")
    for o in obj_info:
        lines.append(f"obj_info {o}")
    lines.append(f"element vertex {n}")
    for p in properties:
        lines.append(f"property {p.ply_type} {p.name}")
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        if fmt == "binary_little_endian":
            arr.tofile(f)
        else:
            for row in arr:
                vals = []
                for p in properties:
                    v = row[p.name]
                    vals.append(repr(float(v)) if p.is_float else str(int(v)))
                f.write((" ".join(vals) + "\n").encode("ascii"))


def gaussian_modeled_properties(table: PlyTable) -> list[str]:
    names = set(table.names)
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY must contain x, y, z properties")
    preferred_prefixes = ("f_dc_", "f_rest_", "scale_", "rot_")
    preferred_exact = {"opacity"}
    picked = [p.name for p in table.properties
              if p.is_float and p.name not in {"x", "y", "z"}
              and (p.name in preferred_exact or p.name.startswith(preferred_prefixes))]
    if picked:
        return picked
    return [p.name for p in table.properties if p.is_float and p.name not in {"x", "y", "z"}]


def raw_coords_from_table(table: PlyTable) -> np.ndarray:
    """Return XYZ without narrowing their stored PLY dtype."""
    cols = [np.asarray(table.data[n]) for n in ("x", "y", "z")]
    dtype = np.result_type(*[c.dtype for c in cols])
    return np.column_stack([c.astype(dtype, copy=False) for c in cols])


def coords_from_table(table: PlyTable) -> np.ndarray:
    """Float32 XYZ used by the compact field fitter."""
    return raw_coords_from_table(table).astype(np.float32, copy=False)


def normalize_coords(coords: np.ndarray, center=None, scale=None):
    coords = np.asarray(coords, dtype=np.float32)
    if center is None:
        center = np.median(coords, axis=0).astype(np.float32)
    if scale is None:
        lo = np.percentile(coords, 1.0, axis=0)
        hi = np.percentile(coords, 99.0, axis=0)
        scale = np.maximum((hi - lo) * 0.5, 1e-6).astype(np.float32)
    return (coords - center[None, :]) / scale[None, :], np.asarray(center, np.float32), np.asarray(scale, np.float32)


def make_freq_matrix(n_rff: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if n_rff <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    directions = rng.normal(size=(n_rff, 3)).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
    scales = np.exp(rng.uniform(np.log(0.5), np.log(8.0), size=(n_rff, 1))).astype(np.float32)
    return directions * scales


def features(q: np.ndarray, freq_matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float32)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    base = np.column_stack([
        np.ones(len(q), np.float32), x, y, z,
        x*x, y*y, z*z, x*y, x*z, y*z,
    ]).astype(np.float32, copy=False)
    if len(freq_matrix) == 0:
        return base
    phase = np.pi * (q @ np.asarray(freq_matrix, np.float32).T)
    return np.concatenate([base, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32, copy=False)


def _sample_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=int(max_samples), replace=False))


def _target_matrix(table: PlyTable, names: Sequence[str], idx) -> np.ndarray:
    return np.column_stack([np.asarray(table.data[name][idx], dtype=np.float32) for name in names])


def fit_field(table: PlyTable, modeled_names: Sequence[str], n_rff: int = 24, ridge: float = 1e-3,
              fit_samples: int = 100_000, seed: int = 0) -> FieldFit:
    if not modeled_names:
        raise ValueError("no modelable floating-point splat attributes found")
    coords = coords_from_table(table)
    q, center, scale = normalize_coords(coords)
    idx = _sample_indices(table.count, fit_samples, seed)
    y = _target_matrix(table, modeled_names, idx)
    mean = y.mean(axis=0).astype(np.float32)
    std = np.maximum(y.std(axis=0).astype(np.float32), 1e-6).astype(np.float32)
    freq = make_freq_matrix(n_rff, seed)
    X = features(q[idx], freq).astype(np.float64)
    yz = ((y - mean[None, :]) / std[None, :]).astype(np.float64)
    xtx = X.T @ X
    reg = float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    reg[0, 0] = 0.0
    W = np.linalg.solve(xtx + reg, X.T @ yz).astype(np.float32)
    return FieldFit(list(modeled_names), mean, std, center, scale, freq, W, float(ridge), int(seed))


def predict_field(fit: FieldFit, coords: np.ndarray) -> np.ndarray:
    q, _, _ = normalize_coords(coords, fit.coord_center, fit.coord_scale)
    X = features(q, fit.freq_matrix)
    yz = X @ fit.weights
    return fit.target_mean[None, :] + yz * fit.target_std[None, :]


def residual_scores(table: PlyTable, fit: FieldFit, chunk: int = 100_000) -> tuple[np.ndarray, dict]:
    coords = coords_from_table(table)
    scores = np.empty(table.count, dtype=np.float32)
    mse_per_attr = np.zeros(len(fit.modeled_names), dtype=np.float64)
    raw_mse_per_attr = np.zeros(len(fit.modeled_names), dtype=np.float64)
    for start in range(0, table.count, chunk):
        stop = min(table.count, start + chunk)
        sl = slice(start, stop)
        y = _target_matrix(table, fit.modeled_names, sl)
        pred = predict_field(fit, coords[sl])
        resid = y - pred
        z = resid / fit.target_std[None, :]
        scores[sl] = np.mean(z*z, axis=1)
        mse_per_attr += np.sum(z*z, axis=0)
        raw_mse_per_attr += np.sum(resid*resid, axis=0)
    mse_per_attr /= max(1, table.count)
    raw_mse_per_attr /= max(1, table.count)
    report = {
        "attribute_nrmse": float(np.sqrt(np.mean(mse_per_attr))),
        "per_attribute_nrmse": {n: float(math.sqrt(v)) for n, v in zip(fit.modeled_names, mse_per_attr)},
        "per_attribute_rmse": {n: float(math.sqrt(v)) for n, v in zip(fit.modeled_names, raw_mse_per_attr)},
    }
    return scores, report


def choose_residual_indices(scores: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("residual fraction must be in [0,1]")
    k = int(round(len(scores) * fraction))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= len(scores):
        return np.arange(len(scores), dtype=np.int64)
    idx = np.argpartition(scores, -k)[-k:]
    return np.sort(idx.astype(np.int64))


def residual_values(table: PlyTable, fit: FieldFit, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros((0, len(fit.modeled_names)), dtype=np.float32)
    coords = coords_from_table(table)[indices]
    y = _target_matrix(table, fit.modeled_names, indices)
    pred = predict_field(fit, coords)
    return (y - pred).astype(np.float32)


def quality_after_residual(scores: np.ndarray, residual_idx: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    if len(residual_idx) == 0:
        return float(np.sqrt(np.mean(scores, dtype=np.float64)))
    mask = np.ones(len(scores), dtype=bool)
    mask[residual_idx] = False
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.sum(scores[mask], dtype=np.float64) / len(scores)))


def _metadata(table: PlyTable, fit: FieldFit, fraction: float, base_report: dict, residual_nrmse: float) -> dict:
    modeled = set(fit.modeled_names)
    return {
        "format": "SplatPack",
        "version": VERSION,
        "source_name": table.path.name,
        "source_ply_format": table.fmt,
        "count": table.count,
        "properties": [{"name": p.name, "ply_type": p.ply_type, "modeled": p.name in modeled} for p in table.properties],
        "comments": table.comments,
        "obj_info": table.obj_info,
        "model": {
            "kind": "ridge-rff-coordinate-field",
            "modeled_names": fit.modeled_names,
            "ridge": fit.ridge,
            "seed": fit.seed,
            "feature_count": fit.feature_count,
            "rff_count": int(len(fit.freq_matrix)),
        },
        "residual_fraction": float(fraction),
        "field_attribute_nrmse": float(base_report["attribute_nrmse"]),
        "packed_attribute_nrmse": float(residual_nrmse),
    }


def write_pack(path: str | Path, table: PlyTable, fit: FieldFit, scores: np.ndarray, base_report: dict,
               fraction: float) -> dict:
    path = Path(path)
    idx = choose_residual_indices(scores, fraction)
    rvals = residual_values(table, fit, idx)
    packed_nrmse = quality_after_residual(scores, idx)
    meta = _metadata(table, fit, fraction, base_report, packed_nrmse)
    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(json.dumps(meta), dtype=np.str_),
        "coords": raw_coords_from_table(table),
        "target_mean": fit.target_mean.astype(np.float32),
        "target_std": fit.target_std.astype(np.float32),
        "coord_center": fit.coord_center.astype(np.float32),
        "coord_scale": fit.coord_scale.astype(np.float32),
        "freq_matrix": fit.freq_matrix.astype(np.float32),
        "weights": fit.weights.astype(np.float32),
        "residual_indices": idx.astype(np.uint32 if table.count < 2**32 else np.uint64),
        "residual_values": rvals.astype(np.float32),
    }
    modeled = set(fit.modeled_names)
    passthrough = [p for p in table.properties if p.name not in modeled and p.name not in {"x", "y", "z"}]
    meta["passthrough"] = []
    for j, p in enumerate(passthrough):
        key = f"raw_{j:03d}"
        arrays[key] = np.asarray(table.data[p.name])
        meta["passthrough"].append({"name": p.name, "key": key})
    arrays["metadata"] = np.asarray(json.dumps(meta), dtype=np.str_)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "residual_splats": int(len(idx)),
        "residual_fraction": float(fraction),
        "packed_attribute_nrmse": packed_nrmse,
    }


def read_pack(path: str | Path):
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["metadata"].item()))
    if meta.get("format") != "SplatPack":
        raise ValueError("not a SplatPack archive")
    return z, meta


def reconstruct_columns(pack_path: str | Path) -> tuple[list[Property], dict[str, np.ndarray], dict]:
    z, meta = read_pack(pack_path)
    props = [Property(x["name"], x["ply_type"]) for x in meta["properties"]]
    coords_raw = np.asarray(z["coords"])
    coords = coords_raw.astype(np.float32, copy=False)
    fit = FieldFit(
        modeled_names=list(meta["model"]["modeled_names"]),
        target_mean=np.asarray(z["target_mean"], np.float32),
        target_std=np.asarray(z["target_std"], np.float32),
        coord_center=np.asarray(z["coord_center"], np.float32),
        coord_scale=np.asarray(z["coord_scale"], np.float32),
        freq_matrix=np.asarray(z["freq_matrix"], np.float32),
        weights=np.asarray(z["weights"], np.float32),
        ridge=float(meta["model"]["ridge"]), seed=int(meta["model"]["seed"]),
    )
    pred = predict_field(fit, coords)
    ridx = np.asarray(z["residual_indices"], dtype=np.int64)
    if len(ridx):
        pred[ridx] += np.asarray(z["residual_values"], dtype=np.float32)
    columns: dict[str, np.ndarray] = {
        "x": coords_raw[:, 0], "y": coords_raw[:, 1], "z": coords_raw[:, 2],
    }
    for j, name in enumerate(fit.modeled_names):
        columns[name] = pred[:, j]
    for item in meta.get("passthrough", []):
        columns[item["name"]] = np.asarray(z[item["key"]])
    return props, columns, meta


def analyze(path: str | Path) -> dict:
    t = read_ply(path)
    modeled = gaussian_modeled_properties(t)
    model_bytes = sum(t.data.dtype.fields[n][0].itemsize for n in modeled) * t.count
    coord_bytes = sum(t.data.dtype.fields[n][0].itemsize for n in ("x", "y", "z")) * t.count
    return {
        "path": str(t.path),
        "format": t.fmt,
        "splats": t.count,
        "properties": len(t.properties),
        "modeled_properties": modeled,
        "modeled_property_count": len(modeled),
        "source_bytes": t.path.stat().st_size,
        "coordinate_bytes": coord_bytes,
        "modelable_attribute_bytes": model_bytes,
        "modelable_fraction_of_source": float(model_bytes / max(1, t.path.stat().st_size)),
    }


def fit_for_pack(path: str | Path, n_rff: int, ridge: float, fit_samples: int, seed: int, chunk: int):
    table = read_ply(path)
    names = gaussian_modeled_properties(table)
    fit = fit_field(table, names, n_rff=n_rff, ridge=ridge, fit_samples=fit_samples, seed=seed)
    scores, base = residual_scores(table, fit, chunk=chunk)
    return table, fit, scores, base


def _print_json(obj):
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_analyze(args):
    _print_json(analyze(args.input))


def cmd_compress(args):
    table, fit, scores, base = fit_for_pack(args.input, args.rff, args.ridge, args.fit_samples, args.seed, args.chunk)
    rec = write_pack(args.output, table, fit, scores, base, args.residual)
    rec.update({
        "source_bytes": table.path.stat().st_size,
        "compression_ratio": float(table.path.stat().st_size / max(1, rec["bytes"])),
        "field_attribute_nrmse": base["attribute_nrmse"],
        "feature_count": fit.feature_count,
        "modeled_properties": len(fit.modeled_names),
    })
    _print_json(rec)


def cmd_decompress(args):
    props, columns, meta = reconstruct_columns(args.input)
    fmt = args.format or meta.get("source_ply_format", "binary_little_endian")
    write_ply(args.output, props, columns, meta.get("comments", []), meta.get("obj_info", []), fmt=fmt)
    _print_json({"output": str(args.output), "bytes": Path(args.output).stat().st_size,
                 "splats": int(meta["count"]), "packed_attribute_nrmse": meta["packed_attribute_nrmse"]})


def cmd_sweep(args):
    table, fit, scores, base = fit_for_pack(args.input, args.rff, args.ridge, args.fit_samples, args.seed, args.chunk)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    stem = Path(args.input).stem
    source_bytes = table.path.stat().st_size
    for frac in args.fractions:
        tag = f"r{frac:.4f}".replace(".", "p")
        p = out / f"{stem}.{tag}.spk"
        rec = write_pack(p, table, fit, scores, base, frac)
        rec["compression_ratio"] = float(source_bytes / max(1, rec["bytes"]))
        rec["source_bytes"] = source_bytes
        rows.append(rec)
        print(f"residual={frac:7.3%}  nrmse={rec['packed_attribute_nrmse']:.6f}  "
              f"size={rec['bytes']/1024:.1f} KiB  ratio={rec['compression_ratio']:.2f}x")
    report = {
        "experiment": "SplatPack rate-distortion sweep",
        "source": str(table.path),
        "splats": table.count,
        "source_bytes": source_bytes,
        "modeled_properties": fit.modeled_names,
        "feature_count": fit.feature_count,
        "field_attribute_nrmse": base["attribute_nrmse"],
        "rows": rows,
    }
    (out / "splatpack_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "splatpack_sweep.csv").open("w", encoding="utf-8") as f:
        f.write("residual_fraction,residual_splats,packed_attribute_nrmse,bytes,compression_ratio\n")
        for r in rows:
            f.write(f"{r['residual_fraction']},{r['residual_splats']},{r['packed_attribute_nrmse']},{r['bytes']},{r['compression_ratio']}\n")
    print(f"wrote {out/'splatpack_sweep.json'}")
    print(f"wrote {out/'splatpack_sweep.csv'}")


def make_demo_ply(path: str | Path, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    x, y, z = xyz.T
    f0 = (0.6*np.sin(2*x) + 0.2*y*z).astype(np.float32)
    f1 = (0.5*np.cos(2.5*y) - 0.25*x*z).astype(np.float32)
    f2 = (0.3*np.sin(2*z + x)).astype(np.float32)
    opacity = (1.2 - 0.8*(x*x+y*y+z*z)).astype(np.float32)
    scale = np.column_stack([-.8+.1*x, -.9+.12*y, -1.0+.08*z]).astype(np.float32)
    rot = np.column_stack([np.ones(n), .08*x, .08*y, .08*z]).astype(np.float32)
    hard = (x > .55) & (y > .35)
    f0[hard] += rng.normal(0, .8, hard.sum()).astype(np.float32)
    f1[hard] += rng.normal(0, .6, hard.sum()).astype(np.float32)
    props = [Property("x","float"), Property("y","float"), Property("z","float"),
             Property("f_dc_0","float"), Property("f_dc_1","float"), Property("f_dc_2","float"),
             Property("opacity","float"),
             Property("scale_0","float"), Property("scale_1","float"), Property("scale_2","float"),
             Property("rot_0","float"), Property("rot_1","float"), Property("rot_2","float"), Property("rot_3","float")]
    cols = {"x":x,"y":y,"z":z,"f_dc_0":f0,"f_dc_1":f1,"f_dc_2":f2,"opacity":opacity,
            "scale_0":scale[:,0],"scale_1":scale[:,1],"scale_2":scale[:,2],
            "rot_0":rot[:,0],"rot_1":rot[:,1],"rot_2":rot[:,2],"rot_3":rot[:,3]}
    write_ply(path, props, cols, comments=["synthetic SplatPack demo"], fmt="binary_little_endian")
    return np.flatnonzero(hard)


def cmd_demo(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ply = out / "demo_gaussians.ply"
    hard = make_demo_ply(ply, n=args.splats, seed=args.seed)
    table, fit, scores, base = fit_for_pack(ply, args.rff, args.ridge, args.fit_samples, args.seed, args.chunk)
    top = choose_residual_indices(scores, min(0.10, max(1/len(scores), len(hard)/len(scores))))
    precision = float(np.mean(np.isin(top, hard))) if len(top) else 0.0
    print(f"demo hard-region top-residual precision={precision:.3f}")
    ns = argparse.Namespace(**vars(args))
    ns.input = str(ply)
    ns.out_dir = str(out / "sweep")
    cmd_sweep(ns)


def cmd_selftest(_args):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "demo.ply"
        make_demo_ply(src, n=600, seed=7)
        a = analyze(src)
        assert a["splats"] == 600 and a["modeled_property_count"] >= 10
        table, fit, scores, base = fit_for_pack(src, 8, 1e-3, 500, 7, 256)
        assert fit.feature_count == 26
        pack = td / "demo.spk"
        rec = write_pack(pack, table, fit, scores, base, 0.1)
        assert pack.exists() and rec["residual_splats"] == 60
        props, cols, meta = reconstruct_columns(pack)
        assert len(cols["x"]) == 600 and len(props) == len(table.properties)
        dst = td / "restored.ply"
        write_ply(dst, props, cols, meta["comments"], meta["obj_info"])
        reread = read_ply(dst)
        assert reread.count == table.count
        z, _ = read_pack(pack)
        ridx = np.asarray(z["residual_indices"], dtype=np.int64)
        for name in fit.modeled_names[:3]:
            err = np.max(np.abs(np.asarray(reread.data[name][ridx], np.float32) - np.asarray(table.data[name][ridx], np.float32)))
            assert err < 2e-5, (name, err)
    print("SplatPack selftest: PASS")


def build_parser():
    ap = argparse.ArgumentParser(prog="splatpack", description="Shared-field + sparse-residual compressor for 3D Gaussian-splat PLY files")
    ap.add_argument("--version", action="version", version=f"SplatPack {VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="inspect a PLY and report modelable attribute bytes")
    p.add_argument("input")
    p.set_defaults(func=cmd_analyze)

    def add_fit_opts(p):
        p.add_argument("--rff", type=int, default=24, help="number of random Fourier frequencies (2 features each)")
        p.add_argument("--ridge", type=float, default=1e-3)
        p.add_argument("--fit-samples", type=int, default=100_000, help="max splats used to fit shared field; <=0 means all")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--chunk", type=int, default=100_000)

    p = sub.add_parser("compress", help="compress PLY to .spk")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--residual", type=float, default=0.05, help="fraction of worst-explained splats stored as exact attribute residuals")
    add_fit_opts(p)
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("decompress", help="reconstruct a PLY from .spk")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--format", choices=["ascii", "binary_little_endian"], default=None)
    p.set_defaults(func=cmd_decompress)

    p = sub.add_parser("sweep", help="build a residual-fraction rate-distortion sweep")
    p.add_argument("input")
    p.add_argument("out_dir")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    add_fit_opts(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("demo", help="make a synthetic Gaussian PLY and run the sweep")
    p.add_argument("out_dir")
    p.add_argument("--splats", type=int, default=4000)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    add_fit_opts(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("selftest", help="run a fast round-trip test")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(f"SplatPack error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
