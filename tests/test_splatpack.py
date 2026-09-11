import tempfile
import unittest
from pathlib import Path

import numpy as np

import splatpack as sp


class SplatPackTests(unittest.TestCase):
    def test_roundtrip_and_exact_residual_rows(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "src.ply"
            sp.make_demo_ply(src, n=500, seed=3)
            table, fit, scores, base = sp.fit_for_pack(src, n_rff=8, ridge=1e-3, fit_samples=400, seed=3, chunk=128)
            out = td / "x.spk"
            rec = sp.write_pack(out, table, fit, scores, base, 0.1)
            self.assertEqual(rec["residual_splats"], 50)
            props, cols, meta = sp.reconstruct_columns(out)
            restored = td / "restored.ply"
            sp.write_ply(restored, props, cols, meta["comments"], meta["obj_info"])
            t2 = sp.read_ply(restored)
            z, _ = sp.read_pack(out)
            ridx = np.asarray(z["residual_indices"], dtype=np.int64)
            for name in fit.modeled_names:
                np.testing.assert_allclose(np.asarray(t2.data[name][ridx]), np.asarray(table.data[name][ridx]), atol=2e-5, rtol=0)

    def test_rate_distortion_is_monotone(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.ply"
            sp.make_demo_ply(src, n=1000, seed=4)
            table, fit, scores, base = sp.fit_for_pack(src, 10, 1e-3, 800, 4, 256)
            qs = []
            for f in [0.0, 0.02, 0.05, 0.10, 0.20]:
                idx = sp.choose_residual_indices(scores, f)
                qs.append(sp.quality_after_residual(scores, idx))
            self.assertTrue(all(a >= b for a, b in zip(qs, qs[1:])), qs)

    def test_ascii_input(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            binp = td / "a.ply"
            sp.make_demo_ply(binp, n=50, seed=1)
            t = sp.read_ply(binp)
            cols = {p.name: np.asarray(t.data[p.name]) for p in t.properties}
            asc = td / "a_ascii.ply"
            sp.write_ply(asc, t.properties, cols, fmt="ascii")
            a = sp.read_ply(asc)
            self.assertEqual(a.count, 50)
            self.assertEqual(a.fmt, "ascii")


if __name__ == "__main__":
    unittest.main()
