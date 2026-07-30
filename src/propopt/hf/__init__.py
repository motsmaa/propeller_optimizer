"""Phase 4: high-fidelity validation path (CalculiX structural; OpenFOAM later).

NOT part of the optimization loop. These solvers are validators run on a handful
of champion designs, never per genome — the reduced models in `solvers/` stay
the inner loop. Requires the `hf` optional extra (gmsh) plus a CalculiX system
binary; see SOURCE.md in this directory for versions and install route.
"""
