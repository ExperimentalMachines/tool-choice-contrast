"""`tcc`: the command surface. Stdlib argparse; every handler returns an exit code."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from toolchoice.io import dumps_artifact, write_json_atomic


def _handle_controls(args: argparse.Namespace, root: Path) -> int:
    from toolchoice.scoring.run_controls import run_all
    from toolchoice.stems.synthetic import population

    report = run_all(population(s_count=args.s_stems, e_count=args.e_stems))
    destination = root / args.out
    if args.verify:
        if not destination.exists():
            print(f"no committed control report at {destination}")
            return 1
        committed = json.loads(destination.read_text(encoding="utf-8"))
        drift = [c["control_id"] for c, p in zip(committed["controls"], report["controls"], strict=True)
                 if c["checks"] != p["checks"]]
        if drift:
            print(f"control outcomes drifted: {', '.join(drift)}")
            return 1
        print(f"VERIFY PASSED - {len(report['controls'])} controls reproduce their committed outcomes")
        return 0

    payload = {k: v for k, v in report.items() if k != "controls"}
    payload["controls"] = [{k: v for k, v in c.items() if k != "scores"} for c in report["controls"]]
    write_json_atomic(destination, payload)
    print(report["render"])
    for control in report["controls"]:
        status = "PASS" if control["passed"] else "FAIL"
        print(f"  {control['control_id']} {status}  {control['description']}")
        for name, ok in control["checks"].items():
            if not ok:
                print(f"      MISS: {name}")
    print(f"\nwrote {destination.relative_to(root)}")
    return 0 if report["all_passed"] else 1


def _handle_stems_validate(args: argparse.Namespace, root: Path) -> int:
    from toolchoice.stems.loader import load_pool, load_stems

    stems_dir, pool_dir = root / args.stems_dir, root / args.tool_pool
    if not any(stems_dir.rglob("*.yaml")):
        print(f"no stems authored under {args.stems_dir}; nothing to validate")
        return 2
    pool = load_pool(pool_dir)
    stems = load_stems(stems_dir, root)
    print(f"{len(stems)} stems, {len(pool)} pool tools")
    return 0


def _handle_population(args: argparse.Namespace, root: Path) -> int:
    from toolchoice.population.build import build, verify

    action = verify if args.verify else build
    return action(root, stems_dir=args.stems_dir, pool_dir=args.tool_pool,
                  out=args.out, manifest=args.manifest)


def _handle_partition(args: argparse.Namespace, root: Path) -> int:
    from toolchoice.population.partition import check_integrity, partition

    population = root / args.population
    if not population.exists():
        print(f"no population at {args.population}")
        return 2
    records = [json.loads(line) for line in population.read_text(encoding="utf-8").splitlines() if line]
    payload = partition(records, pilot_fraction=args.pilot_fraction)
    problems = check_integrity(payload)
    if problems:
        for problem in problems:
            print(f"  {problem}")
        return 1

    destination = root / args.out
    if args.verify:
        if not destination.exists():
            print("VERIFY BLOCKED - no committed partition")
            return 2
        committed = json.loads(destination.read_text(encoding="utf-8"))
        for side in ("pilot", "reported"):
            if committed["sides"][side]["fingerprint"] != payload["sides"][side]["fingerprint"]:
                print(f"VERIFY FAILED - {side} fingerprint moved")
                return 1
        print("VERIFY PASSED - both side fingerprints reproduce")
        return 0

    write_json_atomic(destination, payload)
    for side in ("pilot", "reported"):
        node = payload["sides"][side]
        print(f"  {side:<9} {node['stems']:>4} stems  {node['items']:>4} items  {node['fingerprint'][:16]}")
    print(f"wrote {destination.relative_to(root)}")
    return 0


def _handle_gates(args: argparse.Namespace, root: Path) -> int:
    from toolchoice.verification.gates import main as gates_main

    argv = ["--population", str(args.population), "--minimum-per-mode", str(args.minimum_per_mode)]
    if args.json:
        argv.append("--json")
    return gates_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tcc", description="tool-choice-contrast")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command")

    controls = subparsers.add_parser("controls", help="run the control suite against a synthetic population")
    controls.add_argument("--out", type=Path, default=Path("reports/data/control-suite.json"))
    controls.add_argument("--s-stems", type=int, default=40)
    controls.add_argument("--e-stems", type=int, default=12)
    controls.add_argument("--verify", action="store_true",
                          help="recompute and diff against the committed report")
    controls.set_defaults(handler=_handle_controls)

    stems = subparsers.add_parser("stems", help="authored stems")
    stems_sub = stems.add_subparsers(dest="stems_command")
    validate = stems_sub.add_parser("validate")
    validate.add_argument("--stems-dir", type=Path, default=Path("data/stems"))
    validate.add_argument("--tool-pool", type=Path, default=Path("data/tool_pool"))
    validate.set_defaults(handler=_handle_stems_validate)

    population = subparsers.add_parser("population", help="build or verify the frozen population")
    population_sub = population.add_subparsers(dest="population_command")
    pop_build = population_sub.add_parser("build")
    pop_build.add_argument("--stems-dir", type=Path, default=Path("data/stems"))
    pop_build.add_argument("--tool-pool", type=Path, default=Path("data/tool_pool"))
    pop_build.add_argument("--out", type=Path, default=Path("data/frozen/tcc-v1/population.jsonl"))
    pop_build.add_argument("--manifest", type=Path, default=Path("reports/data/population-manifest.json"))
    pop_build.add_argument("--verify", action="store_true",
                           help="rebuild in memory and diff; writes nothing")
    pop_build.set_defaults(handler=_handle_population)

    pop_part = population_sub.add_parser("partition")
    pop_part.add_argument("--population", type=Path, default=Path("data/frozen/tcc-v1/population.jsonl"))
    pop_part.add_argument("--out", type=Path, default=Path("reports/data/population-partition.json"))
    pop_part.add_argument("--pilot-fraction", type=float, default=0.25)
    pop_part.add_argument("--verify", action="store_true")
    pop_part.set_defaults(handler=_handle_partition)

    gates = subparsers.add_parser("gates", help="run the corpus gates")
    gates.add_argument("--population", type=Path, default=Path("data/frozen/tcc-v1/population.jsonl"))
    gates.add_argument("--minimum-per-mode", type=int, default=150)
    gates.add_argument("--json", action="store_true")
    gates.set_defaults(handler=_handle_gates)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Any = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args, args.root))


__all__ = ["build_parser", "dumps_artifact", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
