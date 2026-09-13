from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "results" / "last_human_miss_audit.json"


def fmt_decision(d):
    o = d["obs"]
    f = d["features"]
    flag = "EXPLORE" if d["exploratory"] else "exploit"
    return (
        f"#{d['decision_index']:>6} {flag:<7} act={d['action']:+.2f} "
        f"eps={d['epsilon']:.3f} margin={d['decision_margin']:.3f} "
        f"evidence={d['base_evidence']:>3}/{d['aim_evidence']:>3} "
        f"ball=({o['ball_x']:.0f},{o['ball_y']:.0f}) "
        f"v=({o['ball_vx']:.0f},{o['ball_vy']:.0f}) speed={o['ball_speed']:.0f} "
        f"paddle_y={o['right_y']:.0f} bins(xf={f['xf']},ef={f['ef']},vf={f['vf']},sp={f['sp']})"
    )


def main():
    if not PATH.exists():
        raise SystemExit("No human miss audit yet. Play once and exit with ESC first.")
    data = json.loads(PATH.read_text(encoding="utf-8"))
    misses = data.get("misses", [])
    print("score:", data.get("final_score"))
    print("miss audits:", len(misses))
    if not misses:
        print("SEM recorded no motor miss in that session.")
        return

    for m in misses:
        print("\n" + "=" * 88)
        print(
            f"MISS #{m['miss_index']}  outcome={m['motor_outcome_index']}  "
            f"likely_cause={m['likely_cause']}"
        )
        print(
            f"intent={m['intent']:+.2f}  decisions={m['num_decisions']}  "
            f"exploratory={m['num_exploratory_decisions']}  max_speed={m['max_ball_speed']:.1f}"
        )
        print("last motor decisions:")
        for d in m.get("last_decisions", []):
            print("  " + fmt_decision(d))


if __name__ == "__main__":
    main()
