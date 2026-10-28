"""Automated structural/symmetry/dimension review for a robot USD.

Complements ``scripts/dump.py``: dump.py prints a full human-readable listing
of bodies/joints/drives; this script runs automated pass/fail/warning checks
that dump.py does not do — schema sanity, left/right naming consistency,
physical left/right symmetry (mass, position, joint limits/drives), and
overall dimension/scale sanity (bounding box, stage units).

Do NOT treat this script's output as ground truth on its own:
- Naming-based left/right pairing is heuristic (substring/prefix matching);
  it can miss real pairs (unusual naming) or invent false pairs.
- Tolerances (--pos-tol, --mass-tol, --expected-height-m) are rough defaults,
  not physical constants for every robot shape (e.g. mark_v3's unicycle will
  not look like a biped and should relax/skip the height check).
Cross-check any FAIL/WARN by also opening the USD in usdview or Isaac Sim, or
by querying the specific prim path directly with a one-off pxr snippet,
before concluding the asset is correct or broken.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any

from isaaclab.app import AppLauncher


@dataclass
class Finding:
    level: str  # "FAIL" | "WARN" | "INFO"
    check: str
    detail: str


def _attr_get(api, getter_name: str):
    if api is None:
        return None
    getter = getattr(api, getter_name, None)
    if getter is None:
        return None
    attr = getter()
    if attr is None:
        return None
    return attr.Get()


def _lr_candidates(name: str) -> list[str]:
    """Plausible left<->right counterpart names for a given prim/joint name."""
    candidates: set[str] = set()
    for a, b in (("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT")):
        if a in name:
            candidates.add(name.replace(a, b))
        if b in name:
            candidates.add(name.replace(b, a))
    # mark_v1-style joint prefix convention: "Lbody_hip" <-> "Rbody_hip"
    if re.match(r"^L[a-z]", name):
        candidates.add("R" + name[1:])
    if re.match(r"^R[a-z]", name):
        candidates.add("L" + name[1:])
    candidates.discard(name)
    return list(candidates)


def _looks_lateral(name: str) -> bool:
    return (
        ("left" in name.lower())
        or ("right" in name.lower())
        or bool(re.match(r"^[LR][a-z]", name))
    )


def _find_pairs(names: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair left/right names. Exact match preferred; case-insensitive match is
    accepted but flagged (this is exactly the class of bug that produced the
    real ``RIght_leg`` vs ``Left_leg`` typo in mark_v1's asset)."""
    name_set = set(names)
    lower_map: dict[str, str] = {}
    for n in names:
        lower_map.setdefault(n.lower(), n)

    pairs: list[tuple[str, str]] = []
    messages: list[str] = []
    seen: set[str] = set()

    for name in names:
        if name in seen or not _looks_lateral(name):
            continue
        candidates = _lr_candidates(name)
        exact = [c for c in candidates if c in name_set and c not in seen]
        if exact:
            other = exact[0]
            pairs.append((name, other))
            seen.add(name)
            seen.add(other)
            continue
        ci_hits = [
            c
            for c in candidates
            if c.lower() in lower_map and lower_map[c.lower()] not in seen
        ]
        if ci_hits:
            other = lower_map[ci_hits[0].lower()]
            pairs.append((name, other))
            seen.add(name)
            seen.add(other)
            messages.append(
                f"NAMING MISMATCH: '{name}' only matches '{other}' case-insensitively "
                f"(expected an exact counterpart like '{ci_hits[0]}') — likely a typo"
            )
            continue
        messages.append(
            f"UNPAIRED: '{name}' looks lateral but has no left/right counterpart"
        )

    return pairs, messages


def review_usd(
    usd_path: str,
    lateral_axis: str = "y",
    pos_tol: float = 0.02,
    mass_tol: float = 0.1,
    limit_tol: float = 0.05,
    expected_height_m: tuple[float, float] | None = (0.3, 2.5),
) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    findings: list[Finding] = []

    def add(level: str, check: str, detail: str) -> None:
        findings.append(Finding(level, check, detail))

    # --- stage units -------------------------------------------------
    try:
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        if abs(meters_per_unit - 1.0) > 1e-6:
            add(
                "WARN",
                "units",
                f"stage metersPerUnit={meters_per_unit} (expected 1.0) — check for a"
                " 100x/cm-vs-m scale mistake before trusting any distances below",
            )
        else:
            add("INFO", "units", "metersPerUnit=1.0")
    except Exception as exc:  # noqa: BLE001 - defensive, this is a diagnostic tool
        add("WARN", "units", f"could not read stage units: {exc}")

    # --- articulation root / bodies / joints --------------------------
    art_roots = [
        p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not art_roots:
        add("FAIL", "articulation_root", "no prim has ArticulationRootAPI")
    elif len(art_roots) > 1:
        add(
            "WARN",
            "articulation_root",
            f"{len(art_roots)} prims have ArticulationRootAPI: {[str(p.GetPath()) for p in art_roots]}",
        )
    else:
        add("INFO", "articulation_root", str(art_roots[0].GetPath()))

    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]

    if not bodies:
        add("FAIL", "rigid_bodies", "no prim has RigidBodyAPI")
    if not joints:
        add("WARN", "joints", "no UsdPhysics.Joint prims found")

    # --- per-body mass + bbox + world position ------------------------
    body_info: dict[str, dict[str, Any]] = {}
    bbox_cache = None
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
    except Exception as exc:  # noqa: BLE001
        add("WARN", "geometry", f"could not create BBoxCache: {exc}")

    for p in bodies:
        name = p.GetName()
        mass = None
        try:
            if p.HasAPI(UsdPhysics.MassAPI):
                mass = _attr_get(UsdPhysics.MassAPI(p), "GetMassAttr")
        except Exception as exc:  # noqa: BLE001
            add("WARN", "mass", f"{p.GetPath()}: MassAPI read failed: {exc}")
        if mass is None or mass <= 0.0:
            add(
                "WARN",
                "mass",
                f"{p.GetPath()}: mass unauthored/zero — solver will substitute a computed"
                " default; verify that's intentional, not a missing MassAPI",
            )

        size = None
        world_pos = None
        if bbox_cache is not None:
            try:
                rng = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
                if not rng.IsEmpty():
                    size = list(rng.GetSize())
                    if min(size) <= 0.0:
                        add(
                            "WARN",
                            "geometry",
                            f"{p.GetPath()}: degenerate bounding box {size}",
                        )
            except Exception as exc:  # noqa: BLE001
                add(
                    "WARN", "geometry", f"{p.GetPath()}: bbox computation failed: {exc}"
                )
        try:
            world_pos = list(
                UsdGeom.Xformable(p)
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                .ExtractTranslation()
            )
        except Exception as exc:  # noqa: BLE001
            add(
                "WARN",
                "geometry",
                f"{p.GetPath()}: world transform failed: {exc}",
            )

        body_info[name] = {
            "path": str(p.GetPath()),
            "mass": mass,
            "size": size,
            "world_pos": world_pos,
        }

    # --- overall dimension sanity --------------------------------------
    if bbox_cache is not None:
        try:
            root_prim = (
                stage.GetDefaultPrim()
                if stage.GetDefaultPrim()
                else stage.GetPseudoRoot()
            )
            overall = bbox_cache.ComputeWorldBound(root_prim).ComputeAlignedRange()
            if not overall.IsEmpty():
                size = list(overall.GetSize())
                add("INFO", "overall_bbox", f"size(x,y,z)={size}")
                height = size[
                    2
                ]  # assumes Z-up (Isaac Sim/USD default); re-check if stage upAxis != Z
                if expected_height_m and not (
                    expected_height_m[0] <= height <= expected_height_m[1]
                ):
                    add(
                        "WARN",
                        "overall_height",
                        f"Z-extent={height:.3f}m outside expected {expected_height_m}m — check stage"
                        " upAxis/scale, or pass --expected-height-m to match this robot's real size"
                        " (e.g. skip/relax for mark_v3's unicycle)",
                    )
        except Exception as exc:  # noqa: BLE001
            add(
                "WARN", "overall_bbox", f"could not compute overall bounding box: {exc}"
            )

    # --- per-joint limits + drive sanity --------------------------------
    joint_info: dict[str, dict[str, Any]] = {}
    for p in joints:
        name = p.GetName()
        is_revolute = p.IsA(UsdPhysics.RevoluteJoint)
        is_prismatic = p.IsA(UsdPhysics.PrismaticJoint)
        lo = hi = stiff = damp = max_force = None
        drive = None
        try:
            if is_revolute:
                rj = UsdPhysics.RevoluteJoint(p)
                lo = _attr_get(rj, "GetLowerLimitAttr")
                hi = _attr_get(rj, "GetUpperLimitAttr")
                drive = UsdPhysics.DriveAPI.Get(p, "angular")
            elif is_prismatic:
                pj = UsdPhysics.PrismaticJoint(p)
                lo = _attr_get(pj, "GetLowerLimitAttr")
                hi = _attr_get(pj, "GetUpperLimitAttr")
                drive = UsdPhysics.DriveAPI.Get(p, "linear")
        except Exception as exc:  # noqa: BLE001
            add("WARN", "joint_limits", f"{name}: failed to read limits: {exc}")

        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if lo > hi:
                add(
                    "FAIL",
                    "joint_limits",
                    f"{name}: lower limit {lo} > upper limit {hi}",
                )
            span = hi - lo
            max_span = 360.0 if is_revolute else 5.0
            if span > max_span:
                unit = "deg" if is_revolute else "m"
                add(
                    "WARN",
                    "joint_limits",
                    f"{name}: range {lo}..{hi}{unit} looks implausibly large for a"
                    f" {'revolute' if is_revolute else 'prismatic'} joint",
                )

        if drive:
            stiff = _attr_get(drive, "GetStiffnessAttr")
            damp = _attr_get(drive, "GetDampingAttr")
            max_force = _attr_get(drive, "GetMaxForceAttr")
            if (stiff in (None, 0.0)) and (damp in (None, 0.0)):
                add(
                    "WARN",
                    "drive",
                    f"{name}: stiffness and damping both 0/unset — GUI-authored drive is"
                    " likely unusable as-is; set PD gains by hand in ImplicitActuatorCfg",
                )
            if max_force in (None, math.inf):
                add(
                    "WARN",
                    "drive",
                    f"{name}: maxForce unset/inf — set effort_limit_sim explicitly",
                )

        joint_info[name] = {
            "type": "revolute"
            if is_revolute
            else ("prismatic" if is_prismatic else str(p.GetTypeName())),
            "lo": lo,
            "hi": hi,
            "stiffness": stiff,
            "damping": damp,
            "max_force": max_force,
        }

    # --- naming + physical left/right symmetry --------------------------
    body_pairs, body_pair_msgs = _find_pairs(list(body_info.keys()))
    joint_pairs, joint_pair_msgs = _find_pairs(list(joint_info.keys()))

    for msg in body_pair_msgs:
        add("WARN" if "MISMATCH" in msg else "INFO", "body_pairing", msg)
    for msg in joint_pair_msgs:
        add("WARN" if "MISMATCH" in msg else "INFO", "joint_pairing", msg)

    lat_idx = {"x": 0, "y": 1, "z": 2}[lateral_axis]
    other_axes = [(ax, i) for ax, i in (("x", 0), ("y", 1), ("z", 2)) if i != lat_idx]

    for left, right in body_pairs:
        li, ri = body_info[left], body_info[right]
        if (
            li["mass"] is not None
            and ri["mass"] is not None
            and max(li["mass"], ri["mass"]) > 0
        ):
            rel = abs(li["mass"] - ri["mass"]) / max(li["mass"], ri["mass"])
            if rel > mass_tol:
                add(
                    "WARN",
                    "symmetry_mass",
                    f"{left}/{right}: mass differs by {rel * 100:.1f}% ({li['mass']} vs {ri['mass']})",
                )
        if li["world_pos"] and ri["world_pos"]:
            lp, rp = li["world_pos"], ri["world_pos"]
            if abs(lp[lat_idx] + rp[lat_idx]) > pos_tol:
                add(
                    "WARN",
                    "symmetry_position",
                    f"{left}/{right}: not mirrored on lateral axis '{lateral_axis}'"
                    f" ({lp[lat_idx]:.3f} vs {rp[lat_idx]:.3f}, expected roughly opposite sign)",
                )
            for ax, idx in other_axes:
                if abs(lp[idx] - rp[idx]) > pos_tol:
                    add(
                        "WARN",
                        "symmetry_position",
                        f"{left}/{right}: '{ax}' positions differ ({lp[idx]:.3f} vs {rp[idx]:.3f})",
                    )

    for left, right in joint_pairs:
        li, ri = joint_info[left], joint_info[right]
        for field_name in ("lo", "hi", "stiffness", "damping"):
            lv, rv = li[field_name], ri[field_name]
            if (
                isinstance(lv, (int, float))
                and isinstance(rv, (int, float))
                and abs(lv - rv) > max(limit_tol, limit_tol * max(abs(lv), abs(rv)))
            ):
                add(
                    "WARN",
                    "symmetry_joint",
                    f"{left}/{right}: '{field_name}' differs ({lv} vs {rv})",
                )

    return {
        "usd_path": usd_path,
        "findings": [asdict(f) for f in findings],
        "body_pairs": body_pairs,
        "joint_pairs": joint_pairs,
        "body_info": body_info,
        "joint_info": joint_info,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"\nUSD REVIEW: {report['usd_path']}")
    by_level: dict[str, list[Finding]] = {"FAIL": [], "WARN": [], "INFO": []}
    for f in report["findings"]:
        by_level[f["level"]].append(f)

    for level in ("FAIL", "WARN", "INFO"):
        items = by_level[level]
        if not items:
            continue
        print(f"\n=== {level} ({len(items)}) ===")
        for f in items:
            print(f"  [{f['check']}] {f['detail']}")

    print("\n=== DETECTED LEFT/RIGHT PAIRS ===")
    print(f"bodies: {report['body_pairs']}")
    print(f"joints: {report['joint_pairs']}")

    n_fail, n_warn = len(by_level["FAIL"]), len(by_level["WARN"])
    print(f"\nSUMMARY: {n_fail} FAIL, {n_warn} WARN. ", end="")
    if n_fail:
        print("Fix FAILs before wiring this asset into an ArticulationCfg.")
    else:
        print(
            "No hard failures — still manually spot-check the WARNs above (see module docstring)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated symmetry/dimension/structural review of a robot USD."
    )
    parser.add_argument(
        "--usd", type=str, default="../assets/mark_v1.usd", help="Path to robot USD."
    )
    parser.add_argument(
        "--lateral-axis",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Left/right mirror axis.",
    )
    parser.add_argument(
        "--pos-tol",
        type=float,
        default=0.02,
        help="Position symmetry tolerance (metres).",
    )
    parser.add_argument(
        "--mass-tol",
        type=float,
        default=0.1,
        help="Relative mass symmetry tolerance (fraction).",
    )
    parser.add_argument(
        "--limit-tol",
        type=float,
        default=0.05,
        help="Relative joint-limit/drive symmetry tolerance.",
    )
    parser.add_argument(
        "--expected-height-m",
        type=float,
        nargs=2,
        default=(0.3, 2.5),
        metavar=("MIN", "MAX"),
        help="Expected overall Z-extent range in metres; use a wide/irrelevant range for non-biped robots (e.g. mark_v3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable report as JSON instead of text.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    report = review_usd(
        args_cli.usd,
        lateral_axis=args_cli.lateral_axis,
        pos_tol=args_cli.pos_tol,
        mass_tol=args_cli.mass_tol,
        limit_tol=args_cli.limit_tol,
        expected_height_m=tuple(args_cli.expected_height_m),
    )

    if args_cli.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        _print_report(report)

    simulation_app.close()


if __name__ == "__main__":
    main()
