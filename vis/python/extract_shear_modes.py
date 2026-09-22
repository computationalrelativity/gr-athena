#!/usr/bin/env python3
"""
extract_shear_modes.py
=======================

Parse an AHF `horizon_shear_<n>.txt` file (written by AHF::Write() in
ahf.cpp) and extract time series of:

  - shear_rms(t)  = sqrt(<sigma_ij sigma^ij>_area)
  - gw_flux(t)    = (1/16pi) * oint sigma_ij sigma^ij dA
  - c_lm(t)       = complex spin-weight -2 coefficients of
                    sigma(theta,phi) = sigma_ab m^a m^b
                                      = sum_{l=2}^{lmax} sum_{m=-l}^{l}
                                        c_lm(t) * _{-2}Y_lm(theta,phi)

File format (as written by AHF::Write() / AHF::SetupIO()):

    # col1: shear_rms = sqrt(<sigma_ij sigma^ij>_area)
    # col2: gw_flux = (1/16pi) * oint sigma_ij sigma^ij dA ...
    # then Re(c_lm) Im(c_lm) pairs for l=2..lmax, m=-l..l, where
    # sigma(theta,phi) = sigma_ab m^a m^b = sum_lm c_lm _{-2}Y_lm(theta,phi)
    # iter = 0, Time = 0
    <shear_rms> <gw_flux> <Re c_{2,-2}> <Im c_{2,-2}> <Re c_{2,-1}> ... <Im c_{lmax,lmax}>
    # iter = 1, Time = 0.5
    <shear_rms> <gw_flux> <Re c_{2,-2}> <Im c_{2,-2}> ...
    ...

lmax is not stored explicitly in the file, so it is inferred from the
number of columns in a data row:

    n_cols   = 2 + 2 * n_modes
    n_modes  = sum_{l=2}^{lmax} (2l+1) = (lmax+1)^2 - 4
    => lmax  = sqrt(n_modes + 4) - 1

Usage
-----
    # Dump everything to a CSV (one row per iteration)
    python extract_shear_modes.py horizon_shear_0.txt -o shear_modes.csv

    # Print a summary and the time series for a couple of modes
    python extract_shear_modes.py horizon_shear_0.txt --modes 2,2 2,0 3,3

    # Quick look plot (amplitude/phase of a mode + shear_rms + gw_flux)
    python extract_shear_modes.py horizon_shear_0.txt --modes 2,2 --plot

As a library:
    from extract_shear_modes import parse_shear_file
    data = parse_shear_file("horizon_shear_0.txt")
    data["time"]        # (nt,) array
    data["shear_rms"]   # (nt,) array
    data["gw_flux"]     # (nt,) array
    data["lmax"]        # int
    data["modes"]       # list of (l, m) tuples, in file column order
    data["c"]           # (nt, n_modes) complex array
    data.mode(2, 2)      # -> (nt,) complex array for l=2, m=2
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Tuple

import numpy as np

ITER_RE = re.compile(r"#\s*iter\s*=\s*(-?\d+)\s*,\s*Time\s*=\s*([^\s]+)")


def _lmax_from_ncols(n_cols: int) -> int:
    """Invert n_cols = 2 + 2*[(lmax+1)^2 - 4] for lmax."""
    n_modes = (n_cols - 2) // 2
    if n_modes < 0 or (n_cols - 2) % 2 != 0:
        raise ValueError(f"Unexpected column count {n_cols} in shear file "
                          f"(expected 2 + 2*n_modes).")
    lmax_sq_arg = n_modes + 4
    lmax = round(sqrt(lmax_sq_arg)) - 1
    if (lmax + 1) ** 2 - 4 != n_modes:
        raise ValueError(f"Column count {n_cols} does not correspond to a "
                          f"consistent l=2..lmax mode set (n_modes={n_modes}).")
    return lmax


def _mode_list(lmax: int) -> List[Tuple[int, int]]:
    """(l, m) pairs in the exact order AHF::Write() emits them:
    l = 2..lmax, m = -l..l."""
    modes = []
    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            modes.append((l, m))
    return modes


@dataclass
class ShearModeData:
    iter: np.ndarray
    time: np.ndarray
    shear_rms: np.ndarray
    gw_flux: np.ndarray
    lmax: int
    modes: List[Tuple[int, int]] = field(default_factory=list)
    c: np.ndarray = None  # complex, shape (nt, n_modes)

    def mode(self, l: int, m: int) -> np.ndarray:
        """Return the complex c_lm(t) time series for a given (l, m)."""
        try:
            idx = self.modes.index((l, m))
        except ValueError:
            raise KeyError(f"Mode (l={l}, m={m}) not present "
                           f"(lmax={self.lmax}).")
        return self.c[:, idx]

    def amplitude(self, l: int, m: int) -> np.ndarray:
        return np.abs(self.mode(l, m))

    def phase(self, l: int, m: int) -> np.ndarray:
        return np.angle(self.mode(l, m))

    def to_csv(self, path: str) -> None:
        header = ["iter", "time", "shear_rms", "gw_flux"]
        cols = [self.iter.astype(float), self.time, self.shear_rms,
                self.gw_flux]
        for (l, m) in self.modes:
            header += [f"Re_c_{l}_{m}", f"Im_c_{l}_{m}"]
        data_re_im = np.empty((self.c.shape[0], 2 * self.c.shape[1]))
        data_re_im[:, 0::2] = self.c.real
        data_re_im[:, 1::2] = self.c.imag
        out = np.column_stack(cols + [data_re_im])
        np.savetxt(path, out, header=" ".join(header), comments="")
        print(f"Wrote {out.shape[0]} rows x {out.shape[1]} cols -> {path}")


def parse_shear_file(path: str, lmax: int | None = None) -> ShearModeData:
    """Parse an AHF horizon_shear_<n>.txt file into a ShearModeData object.

    Parameters
    ----------
    path : str
        Path to horizon_shear_<n>.txt
    lmax : int, optional
        Override the automatically-inferred lmax (normally not needed).
    """
    iters, times, rows = [], [], []
    cur_iter, cur_time = None, None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = ITER_RE.search(line)
                if m:
                    cur_iter = int(m.group(1))
                    cur_time = float(m.group(2))
                continue  # any other comment line (the one-time header)

            vals = [float(x) for x in line.split()]
            rows.append(vals)
            iters.append(cur_iter if cur_iter is not None else len(rows) - 1)
            times.append(cur_time if cur_time is not None else np.nan)

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    n_cols_set = {len(r) for r in rows}
    if len(n_cols_set) != 1:
        raise ValueError(f"Inconsistent column counts in {path}: "
                          f"{sorted(n_cols_set)}. File may be truncated "
                          f"or from a run with a different lmax.")
    n_cols = n_cols_set.pop()

    inferred_lmax = _lmax_from_ncols(n_cols)
    if lmax is not None and lmax != inferred_lmax:
        raise ValueError(f"Requested lmax={lmax} inconsistent with column "
                          f"count {n_cols} (inferred lmax={inferred_lmax}).")
    lmax = inferred_lmax
    modes = _mode_list(lmax)

    arr = np.array(rows, dtype=float)  # (nt, n_cols)
    shear_rms = arr[:, 0]
    gw_flux = arr[:, 1]
    re_im = arr[:, 2:]
    c = re_im[:, 0::2] + 1j * re_im[:, 1::2]  # (nt, n_modes)

    return ShearModeData(
        iter=np.array(iters, dtype=int),
        time=np.array(times, dtype=float),
        shear_rms=shear_rms,
        gw_flux=gw_flux,
        lmax=lmax,
        modes=modes,
        c=c,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_mode_arg(s: str) -> Tuple[int, int]:
    l_str, m_str = s.split(",")
    return int(l_str), int(m_str)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Extract AHF shear-mode time series from "
                    "horizon_shear_<n>.txt")
    p.add_argument("infile", help="path to horizon_shear_<n>.txt")
    p.add_argument("-o", "--csv", metavar="FILE",
                   help="write full time series (shear_rms, gw_flux, all "
                        "c_lm) to a CSV/whitespace-delimited text file")
    p.add_argument("--modes", nargs="+", metavar="l,m",
                   help="specific modes to print/plot, e.g. --modes 2,2 2,0 "
                        "3,3 (default: none -> only print summary)")
    p.add_argument("--lmax", type=int, default=None,
                   help="override auto-detected lmax (sanity check only)")
    p.add_argument("--plot", action="store_true",
                   help="plot shear_rms, gw_flux, and the requested modes "
                        "(requires matplotlib)")
    args = p.parse_args(argv)

    data = parse_shear_file(args.infile, lmax=args.lmax)

    print(f"{args.infile}: {len(data.time)} horizon finds, "
          f"lmax={data.lmax}, {len(data.modes)} modes "
          f"(l=2..{data.lmax})")
    print(f"  time range: [{data.time[0]:.6g}, {data.time[-1]:.6g}]")
    print(f"  shear_rms range: "
          f"[{data.shear_rms.min():.6e}, {data.shear_rms.max():.6e}]")
    print(f"  gw_flux range:   "
          f"[{data.gw_flux.min():.6e}, {data.gw_flux.max():.6e}]")

    requested_modes = [_parse_mode_arg(s) for s in (args.modes or [])]
    for (l, m) in requested_modes:
        c_lm = data.mode(l, m)
        print(f"  mode (l={l}, m={m}): "
              f"|c|_max={np.max(np.abs(c_lm)):.6e} at "
              f"t={data.time[np.argmax(np.abs(c_lm))]:.6g}")

    if args.csv:
        data.to_csv(args.csv)

    if args.plot:
        import matplotlib.pyplot as plt

        n_panels = 2 + len(requested_modes)
        fig, axes = plt.subplots(n_panels, 1, sharex=True,
                                 figsize=(7, 2.2 * n_panels))
        if n_panels == 1:
            axes = [axes]

        axes[0].plot(data.time, data.shear_rms)
        axes[0].set_ylabel(r"shear rms")

        axes[1].plot(data.time, data.gw_flux)
        axes[1].set_ylabel(r"GW flux")

        for k, (l, m) in enumerate(requested_modes):
            c_lm = data.mode(l, m)
            ax = axes[2 + k]
            ax.plot(data.time, c_lm.real, label="Re")
            ax.plot(data.time, c_lm.imag, label="Im")
            ax.plot(data.time, np.abs(c_lm), "k--", label="|c|")
            ax.set_ylabel(f"$c_{{{l},{m}}}$")
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("time")
        fig.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
