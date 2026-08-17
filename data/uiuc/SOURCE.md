# UIUC Propeller Database — raw data files

Propeller: **APC Slow Flyer 10x4.7** (2 blades, diameter 10 in = 0.254 m).
Source: UIUC Propeller Data Site, Volume 1 (J.B. Brandt, R.W. Deters, G.K.
Ananda, O.D. Dantsker, M.S. Selig), https://m-selig.ae.illinois.edu/props/propDB.html

Files (saved unmodified, retrieved 2026-07-11):

| file | contents | URL |
|---|---|---|
| `apcsf_10x4.7_geom.txt` | r/R, c/R, beta (deg, blade angle from rotation plane) | https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_geom.txt |
| `apcsf_10x4.7_static_kt0835.txt` | static (V=0): RPM, CT, CP | https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_static_kt0835.txt |

Advancing-flow (J-sweep) files, retrieved 2026-08-06 for the Phase-4b rung-1
validation. Columns: `J CT CP eta`, at FIXED rpm (the rpm is in the filename).
Both are from the `kt` rig series, the same series as the `kt0835` static file
above, so the static anchor and the advancing anchors are rig-consistent.

| file | rpm | J range | URL |
|---|---|---|---|
| `apcsf_10x4.7_kt0836_4014.txt` | 4014 | 0.142 - 0.719 | https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_kt0836_4014.txt |
| `apcsf_10x4.7_kt0837_5018.txt` | 5018 | 0.115 - 0.576 | https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_kt0837_5018.txt |

`rd`-rig sweeps also exist for this propeller (rd0838_4997, rd0839_6023,
rd0840_6020, rd0841_6512, rd0842_6513) and are NOT downloaded: mixing rigs would
confound a rig offset with a solver error.

Note on rpm choice: CT is weakly rpm-dependent through Reynolds number (the
static table rises 0.1059 -> 0.1299 over 2377 -> 6528 rpm), so a CFD run is
matched to a measured file's rpm EXACTLY rather than interpolated between
sweeps.

Conventions (per the UIUC prop DB documentation): CT = T/(rho n^2 D^4),
CP = P/(rho n^3 D^5) with n in rev/s and D the diameter; beta is the blade
angle measured from the plane of rotation.
