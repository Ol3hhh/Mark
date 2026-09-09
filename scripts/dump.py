"""Dump bodies/joints and properties from a robot USD"""

from __future__ import annotations

import argparse
import math
from math import inf

from isaaclab.app import AppLauncher


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


def _get_drive(joint_prim, axis: str = "angular"):
    from pxr import UsdPhysics

    drive = UsdPhysics.DriveAPI.Get(joint_prim, axis)
    if drive:
        return drive
    return UsdPhysics.DriveAPI.Get(joint_prim, "linear")


def dump_usd(usd_path: str) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    root = stage.GetDefaultPrim()
    bodies: list = []
    joints: list = []
    art_roots: list = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots.append(prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)

    print(f"\nUSD: {usd_path}")
    print(f"Default prim: {root.GetPath() if root and root.IsValid() else '(none)'}")

    print("\n=== ARTICULATION ROOTS ===")
    if not art_roots:
        print("(none found)")
    for p in art_roots:
        print(f"  {p.GetPath()}")

    print("\n=== RIGID BODIES ===")
    print(f"{'idx':>3}  {'prim_path':70}  {'name':30}")
    for i, p in enumerate(bodies):
        print(f"{i:3d}  {p.GetPath()!s:70}  {p.GetName():30}")

    print("\n=== JOINTS (authored in USD) ===")
    print("Revolute lo/hi/target are degrees. Prismatic lo/hi are metres.")
    print(
        f"{'idx':>3}  {'name':28}  {'type':18}  {'lo':>10}  {'hi':>10}  "
        f"{'stiff':>10}  {'damp':>10}  {'max_force':>10}  {'target':>10}"
    )

    def fmt(v) -> str:
        return f"{v:10.4f}" if isinstance(v, (int, float)) else f"{'—':>10}"

    skeleton_rad: dict[str, float] = {}
    drive_unusable = False

    for i, p in enumerate(joints):
        jtype = p.GetTypeName()
        lo = hi = stiff = damp = max_force = target = None
        is_revolute = p.IsA(UsdPhysics.RevoluteJoint)

        if is_revolute:
            rj = UsdPhysics.RevoluteJoint(p)
            lo = _attr_get(rj, "GetLowerLimitAttr")
            hi = _attr_get(rj, "GetUpperLimitAttr")
            drive = _get_drive(p, "angular")
        elif p.IsA(UsdPhysics.PrismaticJoint):
            pj = UsdPhysics.PrismaticJoint(p)
            lo = _attr_get(pj, "GetLowerLimitAttr")
            hi = _attr_get(pj, "GetUpperLimitAttr")
            drive = _get_drive(p, "linear")
        else:
            drive = _get_drive(p, "angular")

        if drive:
            stiff = _attr_get(drive, "GetStiffnessAttr")
            damp = _attr_get(drive, "GetDampingAttr")
            max_force = _attr_get(drive, "GetMaxForceAttr")
            target = _attr_get(drive, "GetTargetPositionAttr")

        print(
            f"{i:3d}  {p.GetName():28}  {jtype:18}  {fmt(lo)}  {fmt(hi)}  "
            f"{fmt(stiff)}  {fmt(damp)}  {fmt(max_force)}  {fmt(target)}"
        )

        if stiff == 0.0 or damp == 0.0 or max_force in (None, inf):
            drive_unusable = True

        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            mid = 0.5 * (float(lo) + float(hi))
            skeleton_rad[p.GetName()] = math.radians(mid) if is_revolute else mid
        elif isinstance(target, (int, float)):
            skeleton_rad[p.GetName()] = (
                math.radians(float(target)) if is_revolute else float(target)
            )
        else:
            skeleton_rad[p.GetName()] = 0.0

    print("\n=== suggested init_state.joint_pos (mid-limits in radians) ===")
    print("joint_pos={")
    for name, val in skeleton_rad.items():
        print(f'    "{name}": {val:.6f},')
    print("}")
    print(
        "\nMid-limits are a kinematic centre, not a standing pose. "
        "Paste names into robots/mark_v1.py; convert USD degrees with radians() "
        "and tune stance / PD in Sim."
    )
    if drive_unusable:
        print(
            "USD drive is stiffness=0 / damping=0 / maxForce=inf — "
            "do not copy into ImplicitActuatorCfg; set PD and effort_limit_sim by hand."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump joint/body info from a USD file."
    )
    parser.add_argument(
        "--usd",
        type=str,
        default="../assets/mark_v1.usd",
        help="Path to robot USD.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    dump_usd(args_cli.usd)
    simulation_app.close()


if __name__ == "__main__":
    main()
