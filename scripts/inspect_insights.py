"""Quick console summary of insights.json (for verification and demos).

    python scripts/inspect_insights.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data = json.loads(
    (PROJECT_ROOT / "data/processed/insights.json").read_text(encoding="utf-8")
)

print("corpus:", data["corpus"])
print("headline:", data["executive_summary"]["headline"])
ok = sum(1 for i in data["insights"] if i["citation_check_passed"])
print(f"citations verified: {ok}/{len(data['insights'])}")
badges = {}
for i in data["insights"]:
    badges[i["confidence"]["badge"]] = badges.get(i["confidence"]["badge"], 0) + 1
print("confidence badges:", badges)

print("\nTOP 8 BY PRIORITY:")
for i in data["insights"][:8]:
    a = i["analysis"]
    c = i["confidence"]
    print(
        f"  [{i['priority_score']:.2f}] {a['theme_name']} "
        f"(sev {a['severity']}/5, n={c['n_items']}, badge={c['badge']}, "
        f"sources={i['per_source_counts']})"
    )
    print(f"      root cause: {a['root_causes'][0][:110]}")
    print(f"      action:     {a['recommended_actions'][0]['action'][:110]}")
