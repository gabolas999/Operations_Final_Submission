"""Utilities to reuse MILP.py reporting/plotting with Meta_Heuristics solutions.

The plotting and reporting methods in [MILP.py](MILP.py) expect Gurobi-style
variables that expose a `.X` attribute (e.g., `x_ijk[i,j,k].X`).

Your meta-heuristic produces an assignment matrix `f_ck` (containers -> barges)
plus can reconstruct per-barge routes and timings via `get_route`/`get_timing`.

This module builds a lightweight "view" by attaching variable-like containers
onto an existing `MILP_Algo` instance, so you can call MILP methods like:

- `print_results_2()`
- `print_barge_table()`
- `print_time_schedule()`
- `plot_barge_solution_map_report_3()`
- `plot_time_windows()`

without solving the MILP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

import numpy as np


@dataclass
class _Var:
    """Minimal stand-in for a Gurobi Var exposing `.X`."""

    X: float


class _VarDict(dict):
    """Dict that returns `_Var(0.0)` for missing keys.

    This mimics how an unselected Gurobi variable would behave (`.X == 0`).
    """

    def __getitem__(self, key):  # type: ignore[override]
        return dict.get(self, key, _Var(0.0))


@dataclass
class _FakeModel:
    """Minimal model object to satisfy MILP plotting guards."""

    status: int


def attach_meta_solution_as_milp_view(
    milp_instance,
    *,
    meta_f_ck: np.ndarray,
    get_route: Callable[[Mapping[int, dict]], List[int]],
    get_timing: Callable[
        [List[int], Mapping[int, dict], float], Tuple[List[float], List[float]]
    ],
    departure_delay_hours: float = 0.0,
):
    """Attach meta-heuristic results onto a `MILP_Algo` instance.

    Parameters
    - `milp_instance`: an instance of `MILP_Algo` that already has instance data
      (`C_dict`, `T_ij_matrix`, `W_c`, `E/I`, `Z_cj`, `node_xy`, etc.).
    - `meta_f_ck`: shape (C, K_b) assignment matrix (1 if container c on barge k).
      Containers with all zeros are treated as trucked.
    - `get_route(Lcur)`: returns a route like `[0, j1, j2, ..., 0]` for a given barge.
    - `get_timing(route, Lcur, delay)`: should return `(D_term, O_term)` where
      `O_term[idx]` is the arrival/start-of-service time at `route[idx]`.

    Returns
    - The same `milp_instance`, mutated with `.model`, `.f_ck`, `.x_ijk`, `.y_ijk`,
      `.z_ijk`, and `.t_jk` so that MILP reporting/plotting works.
    """

    # Import here so users can still import this module without Gurobi installed,
    # as long as they don't call this function.
    from gurobipy import GRB

    C = milp_instance.C_list
    N = milp_instance.N_list
    K_b = milp_instance.K_b
    K_t = milp_instance.K_t

    if meta_f_ck.shape != (len(C), len(K_b)):
        raise ValueError(
            "meta_f_ck must have shape (C, K_b). "
            f"Got {meta_f_ck.shape}, expected ({len(C)}, {len(K_b)})."
        )

    # --------------- f_ck (containers -> vehicles, incl truck) ---------------
    f_ck = _VarDict()
    for c in C:
        assigned_any = False
        for k in K_b:
            if float(meta_f_ck[c, k]) > 0.5:
                f_ck[(c, k)] = _Var(1.0)
                assigned_any = True
        if not assigned_any:
            f_ck[(c, K_t)] = _Var(1.0)

    # --------------- x_ijk, y_ijk, z_ijk, t_jk from reconstructed routes ---------------
    x_ijk = _VarDict()
    y_ijk = _VarDict()
    z_ijk = _VarDict()
    t_jk = _VarDict()

    C_dict = milp_instance.C_dict

    for k in K_b:
        assigned = [c for c in C if f_ck[(c, k)].X > 0.5]
        if not assigned:
            continue

        Lcur = {c: C_dict[c] for c in assigned}
        route = get_route(Lcur)
        if not route or route[0] != 0:
            raise ValueError(
                f"get_route must return a route starting at 0. Got: {route}"
            )

        # Ensure route ends at depot for downstream assumptions.
        if route[-1] != 0:
            route = list(route) + [0]

        # Arrival/service times per node on the route.
        _, O_term = get_timing(route, Lcur, float(departure_delay_hours))
        if len(O_term) != len(route):
            raise ValueError(
                "get_timing must return arrays aligned with route. "
                f"len(route)={len(route)} but len(O_term)={len(O_term)}"
            )

        for idx, node in enumerate(route):
            t_jk[(node, k)] = _Var(float(O_term[idx]))

        # TEU flow split into "exports remaining" and "imports onboard".
        export_remaining = sum(
            info["Wc"] for info in Lcur.values() if info["In_or_Out"] == 2
        )
        import_onboard = 0

        for i, j in zip(route[:-1], route[1:]):
            if i == j:
                continue

            x_ijk[(i, j, k)] = _Var(1.0)
            z_ijk[(i, j, k)] = _Var(float(export_remaining))
            y_ijk[(i, j, k)] = _Var(float(import_onboard))

            # Update onboard split when arriving at j.
            if j != 0:
                exports_unloaded = sum(
                    info["Wc"]
                    for info in Lcur.values()
                    if info["Terminal"] == j and info["In_or_Out"] == 2
                )
                imports_loaded = sum(
                    info["Wc"]
                    for info in Lcur.values()
                    if info["Terminal"] == j and info["In_or_Out"] == 1
                )
                export_remaining -= exports_unloaded
                import_onboard += imports_loaded
            else:
                # Back at depot: unload imports.
                import_onboard = 0

    # Attach onto instance
    milp_instance.model = _FakeModel(status=GRB.OPTIMAL)
    milp_instance.f_ck = f_ck
    milp_instance.x_ijk = x_ijk
    milp_instance.y_ijk = y_ijk
    milp_instance.z_ijk = z_ijk
    milp_instance.t_jk = t_jk

    return milp_instance
