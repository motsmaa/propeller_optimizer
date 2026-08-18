"""Azimuthally-averaged through-disc inflow from a converged rotor CFD field.

Extracted from `rung0_reprice.py` so the rung-0 reprice and the 2-D section
study read the inflow through ONE seam rather than two copies that can drift.

FRAME (from hover_case.py): x tangential, y AXIAL (rotation axis), z span.
Blades lie along +-z. Thrust acts along +y and the flow through the disc goes
-y, so the through-disc speed is v_ax = -Uy (positive when flowing through).
Rotation is about +y with omega NEGATIVE, so the blade-travel unit vector at
(x,y,z) is t = (-z,0,x)/r and the fluid's co-rotating swirl is
u_sw = U.t = (-z*Ux + x*Uz)/r.

BLADE CONTAMINATION. The disc plane y=0 IS the blade plane: a radial ray at a
blade azimuth runs inside the solid. Points inside the blade are simply absent
from the sample (no cell to interpolate), but points just OFF the surface carry
the blade's own bound-vortex up/downwash — the self-induced part that
blade-element theory accounts for through the lift itself, and which must NOT be
counted as inflow. Azimuths within `blade_excl_deg` of a blade are dropped.
"""
from __future__ import annotations

import re

import numpy as np

from ..contracts import SolverError
from .openfoam import _wsl_sh, run_foam
from .wsl import _quote, to_wsl_path

BLADE_AZ = (90.0, 270.0)        # blades lie along +-z


def write_disc_sampling(src, work, n_az=36, az_off=5.0, n_rad=80,
                        r0f=0.10, r1f=1.05, planes=(0.0,), radius=0.127):
    """Rewrite the case controlDict with radial rays at several axial planes.

    `axis xyz` so every sampled point carries its own coordinates: the radius
    and azimuth are then computed from the DATA, not assumed from the ray index,
    which keeps the reader correct even when snappy has dropped points.
    """
    az = np.radians(az_off + 360.0 * np.arange(n_az) / n_az)
    sets = []
    for iy, yf in enumerate(planes):
        for ia, a in enumerate(az):
            r0, r1 = r0f * radius, r1f * radius
            s = (r0 * np.cos(a), yf * radius, r0 * np.sin(a))
            e = (r1 * np.cos(a), yf * radius, r1 * np.sin(a))
            sets.append("        s%d_%02d { type uniform; axis xyz; start "
                        "(%.6f %.6f %.6f); end (%.6f %.6f %.6f); nPoints %d; }"
                        % (iy, ia, s[0], s[1], s[2], e[0], e[1], e[2], n_rad))
    fn = "\n".join(["functions", "{", "    discs { type sets; libs (sampling);",
                    "      setFormat raw; interpolationScheme cellPoint;",
                    "      fields (U p); writeControl writeTime;",
                    "      sets", "      {"] + sets + ["      } }", "}", ""])
    cd = src / "system" / "controlDict"
    cd.write_text(re.sub(r"functions\s*\{.*", fn, cd.read_text(), flags=re.S))
    _wsl_sh("cp " + _quote(to_wsl_path(cd)) + " " + _quote(work)
            + "/system/controlDict")


def read_disc_samples(case, name=""):
    """Parse the sampled rays, keyed by plane index."""
    sdir = _wsl_sh("cd " + _quote(case.work) + " && ls -d postProcessing/discs/* "
                   "2>/dev/null | tail -1").stdout.strip()
    if not sdir:
        raise SolverError(name + ": no postProcessing/discs output")
    raw = _wsl_sh("cd " + _quote(case.work) + "/" + sdir
                  + " && for f in s*_p_U.xy; do awk -v n=\"$f\" "
                    "'!/^#/ && NF>0 {print n, $0}' \"$f\"; done")
    if raw.returncode != 0 or not raw.stdout.strip():
        raise SolverError(name + ": sample files unreadable: " + raw.stderr[:300])
    rows = {}
    for ln in raw.stdout.splitlines():
        p = ln.split()
        if len(p) != 8:                 # name + x y z p Ux Uy Uz
            raise SolverError(name + ": expected 7 data columns "
                              "(x y z p Ux Uy Uz), got %d in %r"
                              % (len(p) - 1, ln[:120]))
        m = re.match(r"s(\d+)_(\d+)_p_U\.xy", p[0])
        rows.setdefault(int(m.group(1)), []).append([float(v) for v in p[1:]])
    return rows


def azimuthal_profile(rows, iy, rg, exclude_blades=True, blade_excl_deg=20.0):
    """Azimuthally averaged (v_ax, u_swirl) on the `rg` radial grid.
    Returns (mean[2,n], median[2,n], n_used[n])."""
    a = np.asarray(rows[iy])
    x, z, ux, uy, uz = a[:, 0], a[:, 2], a[:, 4], a[:, 5], a[:, 6]
    r = np.hypot(x, z)
    ok = r > 1e-9
    psi = np.degrees(np.arctan2(z[ok], x[ok])) % 360.0
    if exclude_blades:
        d = np.minimum.reduce([np.abs((psi - b + 180.0) % 360.0 - 180.0)
                               for b in BLADE_AZ])
        ok[np.flatnonzero(ok)[d < blade_excl_deg]] = False
    r, x, z = r[ok], x[ok], z[ok]
    v_ax = -uy[ok]                                  # +ve = through the disc
    u_sw = (-z * ux[ok] + x * uz[ok]) / r           # +ve = co-rotating
    mean = np.full((2, rg.size), np.nan)
    med = np.full((2, rg.size), np.nan)
    n = np.zeros(rg.size, dtype=int)
    half = 0.5 * (rg[1] - rg[0])
    for i, rc in enumerate(rg):
        m = np.abs(r - rc) <= half
        n[i] = int(m.sum())
        if n[i]:
            mean[:, i] = [v_ax[m].mean(), u_sw[m].mean()]
            med[:, i] = [np.median(v_ax[m]), np.median(u_sw[m])]
    return mean, med, n


def latest_time(case, name=""):
    t = _wsl_sh("cd " + _quote(case.work)
                + " && ls -d [0-9]* 2>/dev/null | sort -g | tail -1").stdout.strip()
    if not t or t == "0":
        raise SolverError(name + ": no reconstructed time directory (got "
                          + repr(t) + "); refusing to sample")
    return t


def run_sampling(case, src, name="", **kw):
    """Write the sampling dict and run the post-processor. Returns parsed rows."""
    write_disc_sampling(src, case.work, **kw)
    run_foam("simpleFoam", case, ("-postProcess", "-latestTime"),
             timeout=14400, log_name="log.discsample")
    return read_disc_samples(case, name)
