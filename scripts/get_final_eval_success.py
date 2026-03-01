#!/usr/bin/env python3
"""Read all evaluator logs and output final success scores as a table."""
import csv
import os
import sys

def main():
    results = []
    for root, dirs, files in os.walk("logs"):
        for f in files:
            if f != "logs.csv" or "evaluator" not in root:
                continue
            path = os.path.join(root, f)
            parts = path.replace("\\", "/").split("/")
            if len(parts) >= 2 and parts[0] == "logs":
                run_name = parts[1]
            else:
                run_name = path
            try:
                with open(path) as fp:
                    reader = csv.DictReader(fp)
                    rows = list(reader)
                    if not rows:
                        continue
                    last = rows[-1]
                    success_1000 = last.get("success_1000", "")
                    success = last.get("success", "")
                    try:
                        s1 = float(success_1000) if success_1000 and success_1000 != "nan" else None
                    except ValueError:
                        s1 = None
                    try:
                        s = float(success) if success else None
                    except ValueError:
                        s = None
                    final = s1 if (s1 is not None and 0 <= s1 <= 1) else s
                    if final is None and s is not None:
                        final = s
                    toks = run_name.split("_")
                    if len(toks) >= 4 and toks[0] == "contrastive" and toks[1] == "cpc":
                        seed = toks[-1]
                        env = "_".join(toks[2:-1])
                    else:
                        env = run_name
                        seed = ""
                    results.append({"env": env, "seed": seed, "run": run_name, "final_success": final})
            except Exception as e:
                results.append({"env": run_name, "seed": "", "run": run_name, "final_success": None, "err": str(e)})

    # Sort by env, seed (show all runs; multiple UUIDs for same env_seed appear as separate rows)
    results.sort(key=lambda x: (x["env"], x["seed"]))

    # Markdown table
    lines = [
        "| Environment | Seed | Final evaluation success |",
        "|-------------|------|--------------------------|",
    ]
    for r in results:
        fs = r.get("final_success")
        if fs is not None:
            lines.append("| {} | {} | {:.2%} |".format(r["env"], r["seed"], fs))
        else:
            lines.append("| {} | {} | {} |".format(r["env"], r["seed"], r.get("err", "N/A")))
    text = "\n".join(lines)
    print(text)
    return text

if __name__ == "__main__":
    main()
