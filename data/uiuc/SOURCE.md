# UIUC Propeller Database — raw data files

Propeller: **APC Slow Flyer 10x4.7** (2 blades, diameter 10 in = 0.254 m).
Source: UIUC Propeller Data Site, Volume 1 (J.B. Brandt, R.W. Deters, G.K.
Ananda, O.D. Dantsker, M.S. Selig), https://m-selig.ae.illinois.edu/props/propDB.html

Files (saved unmodified, retrieved 2026-07-11):

| file | contents | URL |
|---|---|---|
| `apcsf_10x4.7_geom.txt` | r/R, c/R, beta (deg, blade angle from rotation plane) | https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_geom.txt |
| `apcsf_10x4.7_static_kt0835.txt` | static (V=0): RPM, CT, CP | https://m-selig.ae.illinois.edu/props/volume-1/data/apcsf_10x4.7_static_kt0835.txt |

Conventions (per the UIUC prop DB documentation): CT = T/(rho n^2 D^4),
CP = P/(rho n^3 D^5) with n in rev/s and D the diameter; beta is the blade
angle measured from the plane of rotation.
