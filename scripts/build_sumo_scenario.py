"""Build SUMO highway accident scenario files from configs/env.yaml.

Regenerates:
  sumo/highway.nod.xml
  sumo/highway.edg.xml
  sumo/highway.rou.xml
  sumo/highway.add.xml
  sumo/highway.net.xml   (via netconvert)
  sumo/highway.sumocfg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "env.yaml"
SUMO_DIR = PROJECT_ROOT / "sumo"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_xml(path: Path, root: ET.Element) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def build_nodes(cfg: dict) -> Path:
    hw = cfg["highway"]
    root = ET.Element("nodes")
    ET.SubElement(
        root,
        "node",
        id=str(hw["from_node"]),
        x="0.00",
        y="0.00",
        type="priority",
    )
    ET.SubElement(
        root,
        "node",
        id=str(hw["to_node"]),
        x=f"{float(hw['length_m']):.2f}",
        y="0.00",
        type="priority",
    )
    out = SUMO_DIR / "highway.nod.xml"
    write_xml(out, root)
    return out


def build_edges(cfg: dict) -> Path:
    hw = cfg["highway"]
    root = ET.Element("edges")
    edge = ET.SubElement(root, "edge")
    edge.set("id", str(hw["edge_id"]))
    edge.set("from", str(hw["from_node"]))
    edge.set("to", str(hw["to_node"]))
    edge.set("numLanes", str(int(hw["lanes"])))
    edge.set("speed", f"{float(hw['speed_limit_mps']):.2f}")
    edge.set("length", f"{float(hw['length_m']):.2f}")

    out = SUMO_DIR / "highway.edg.xml"
    write_xml(out, root)
    return out


def _vtype_attrs(type_id: str, params: dict) -> dict[str, str]:
    attrs = {
        "id": type_id,
        "accel": str(params["accel"]),
        "decel": str(params["decel"]),
        "sigma": str(params["sigma"]),
        "length": str(params["length"]),
        "minGap": str(params["min_gap"]),
        "maxSpeed": str(params["max_speed"]),
        "color": str(params["color"]),
        "carFollowModel": "Krauss",
    }
    # GUI only: passenger body instead of default triangle. No dynamics change.
    if params.get("gui_shape"):
        attrs["guiShape"] = str(params["gui_shape"])
    return attrs


def build_routes(cfg: dict) -> Path:
    hw = cfg["highway"]
    traffic = cfg["traffic"]
    accident = cfg["accident"]
    vtypes = cfg["vehicle_types"]

    total = float(traffic["vehs_per_hour"])
    cav_p = float(traffic["cav_penetration"])
    if not 0.0 <= cav_p <= 1.0:
        raise ValueError(f"cav_penetration must be in [0, 1], got {cav_p}")

    cav_vph = int(round(total * cav_p))
    hdv_vph = int(round(total - cav_vph))
    # Keep at least one of each when penetration is interior (for TraCI tests).
    if 0.0 < cav_p < 1.0:
        cav_vph = max(1, cav_vph)
        hdv_vph = max(1, int(round(total - cav_vph)))

    edge_id = str(hw["edge_id"])
    n_lanes = int(hw["lanes"])
    blocked = [int(x) for x in accident["blocked_lane_indices"]]
    if not blocked:
        raise ValueError("accident.blocked_lane_indices must not be empty")
    if any(i < 0 or i >= n_lanes for i in blocked):
        raise ValueError(f"blocked lanes {blocked} out of range for {n_lanes} lanes")
    if len(blocked) >= n_lanes:
        raise ValueError("must leave at least one open lane")

    pos = float(accident["position_m"])
    half = float(accident["stop_half_width_m"])
    stop_start = max(0.0, pos - half)
    stop_end = pos + half
    prefix = str(accident.get("vehicle_id_prefix", "ACCIDENT"))

    root = ET.Element("routes")
    ET.SubElement(root, "vType", **_vtype_attrs("HDV", vtypes["HDV"]))
    ET.SubElement(root, "vType", **_vtype_attrs("CAV", vtypes["CAV"]))
    ET.SubElement(root, "vType", **_vtype_attrs("ACCIDENT", vtypes["ACCIDENT"]))
    ET.SubElement(root, "route", id="main", edges=edge_id)

    flow_common = {
        "route": "main",
        "begin": str(traffic["flow_begin_s"]),
        "end": str(traffic["flow_end_s"]),
        "departLane": str(traffic["depart_lane"]),
        "departSpeed": str(traffic["depart_speed"]),
    }
    ET.SubElement(
        root,
        "flow",
        id="flow_HDV",
        type="HDV",
        vehsPerHour=str(hdv_vph),
        **flow_common,
    )
    ET.SubElement(
        root,
        "flow",
        id="flow_CAV",
        type="CAV",
        vehsPerHour=str(cav_vph),
        **flow_common,
    )

    for lane_idx in blocked:
        veh = ET.SubElement(
            root,
            "vehicle",
            id=f"{prefix}_{lane_idx}",
            type="ACCIDENT",
            route="main",
            depart="0",
            departLane=str(lane_idx),
            departPos=f"{pos:.2f}",
            departSpeed="0",
        )
        ET.SubElement(
            veh,
            "stop",
            lane=f"{edge_id}_{lane_idx}",
            startPos=f"{stop_start:.2f}",
            endPos=f"{stop_end:.2f}",
            duration=str(accident["duration_s"]),
        )

    out = SUMO_DIR / "highway.rou.xml"
    write_xml(out, root)
    return out


def build_additional(cfg: dict) -> Path:
    """Reserved additional file (no POI / detectors — keeps GUI lanes uncolored)."""
    del cfg  # config reserved for future additional objects
    root = ET.Element("additional")
    out = SUMO_DIR / "highway.add.xml"
    write_xml(out, root)
    return out


def build_sumocfg(cfg: dict) -> Path:
    sumo = cfg["sumo"]
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value="highway.net.xml")
    ET.SubElement(inp, "route-files", value="highway.rou.xml")
    ET.SubElement(inp, "additional-files", value="highway.add.xml")

    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin", value=str(sumo["begin_s"]))
    ET.SubElement(time, "end", value=str(sumo["end_s"]))
    ET.SubElement(time, "step-length", value=str(sumo["step_length"]))

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "collision.action", value=str(sumo["collision_action"]))
    ET.SubElement(proc, "time-to-teleport", value=str(sumo["time_to_teleport"]))

    rnd = ET.SubElement(root, "random_number")
    ET.SubElement(rnd, "seed", value=str(cfg.get("seed", 42)))

    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-warnings", value="false")
    ET.SubElement(report, "no-step-log", value="true")

    # GUI-only: enable "simple shapes" so guiShape="passenger" is drawn.
    gui = ET.SubElement(root, "gui_only")
    ET.SubElement(gui, "gui-settings-file", value="highway.gui.xml")

    out = SUMO_DIR / "highway.sumocfg"
    write_xml(out, root)
    return out


def run_netconvert(nod: Path, edg: Path, net_out: Path) -> None:
    cmd = [
        "netconvert",
        f"--node-files={nod}",
        f"--edge-files={edg}",
        f"--output-file={net_out}",
        "--no-turnarounds", "true",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"netconvert failed with code {result.returncode}")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def build_all(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    SUMO_DIR.mkdir(parents=True, exist_ok=True)

    nod = build_nodes(cfg)
    edg = build_edges(cfg)
    rou = build_routes(cfg)
    add = build_additional(cfg)
    cfg_out = build_sumocfg(cfg)
    net = SUMO_DIR / "highway.net.xml"
    run_netconvert(nod, edg, net)

    traffic = cfg["traffic"]
    total = float(traffic["vehs_per_hour"])
    cav_p = float(traffic["cav_penetration"])
    cav_vph = max(1, int(round(total * cav_p))) if 0 < cav_p < 1 else int(round(total * cav_p))
    hdv_vph = max(1, int(round(total - cav_vph))) if 0 < cav_p < 1 else int(round(total - cav_vph))

    print("Built SUMO scenario:")
    print(f"  {nod}")
    print(f"  {edg}")
    print(f"  {rou}")
    print(f"  {add}")
    print(f"  {net}")
    print(f"  {cfg_out}")
    print(
        f"Traffic: total={total:.0f} veh/h, CAV penetration={cav_p:.2f} "
        f"(HDV={hdv_vph}, CAV={cav_vph})"
    )
    print(
        f"Accident: lanes={cfg['accident']['blocked_lane_indices']} "
        f"pos={cfg['accident']['position_m']} m"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 1 SUMO scenario")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to env.yaml",
    )
    args = parser.parse_args()
    build_all(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
