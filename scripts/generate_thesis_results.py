"""
generate_thesis_results.py
==========================
Reads SUMO tripinfo XML files from batch_results subfolders across three study
sites and produces publication-quality bar charts comparing DRL agents vs the
No-DRL baseline.

Metrics extracted from *_trips_*.xml (per vehicle / per person):
  - avg_waiting_time   : mean vehicle waiting time (s)  — lower is better for DRL
  - avg_time_loss      : mean vehicle time loss (s)     — lower is better for DRL
  - throughput         : vehicles that arrived          — higher is better for DRL
  - avg_travel_time    : mean trip duration (s)         — lower is better for DRL
  - ped_waiting_time   : mean pedestrian waiting time   — lower is better for DRL
  - ped_time_loss      : mean pedestrian time loss      — lower is better for DRL

Grouping:
  SP_*  = Signalized Pedestrian (with DRL or NoDRL)
  BP_*  = Baseline Pedestrian   (with DRL or NoDRL)

For each location one figure is produced per metric, with:
  X-axis : scenario (Normal / Slow / Heavy Traffic)
  Bars   : one per agent, grouped by scenario
  Error  : 95% confidence interval across runs

Usage (from repo root):
    python scripts/generate_thesis_results.py
      --root   .                  # repo root
      --outdir results_figs       # output folder
      --prefix SP                 # SP or BP or ALL
"""
from pathlib import Path
import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import math

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import stats as scipy_stats
except Exception as e:
    raise SystemExit(
        f"Missing dependency ({e}). Run:\n"
        "  pip install pandas numpy matplotlib seaborn scipy openpyxl"
    )


# ── Config ────────────────────────────────────────────────────────────────────

LOCATIONS = [
    ("Olivarez_traci",       "Olivarez College"),
    ("Banlic-Mamatid_traci", "Banlic-Mamatid"),
    ("Balibago_traci",       "Balibago"),
]

SCENARIO_ORDER  = ["normal", "slow", "jam"]
SCENARIO_LABELS = {"normal": "Normal", "slow": "Slow", "jam": "Heavy Traffic"}

# Metrics to plot: (column_name, display_label, higher_is_better)
METRICS = [
    ("avg_waiting_time", "Avg. Waiting Time (s)",     False),
    ("avg_time_loss",    "Avg. Time Loss (s)",         False),
    ("throughput",       "Throughput (vehicles arrived)", True),
    ("avg_travel_time",  "Avg. Travel Time (s)",       False),
    ("ped_waiting_time", "Pedestrian Avg. Waiting Time (s)", False),
    ("ped_time_loss",    "Pedestrian Avg. Time Loss (s)",    False),
]

# Pretty names for agents in legend
AGENT_LABELS = {
    "SP_NoDRL": "No-DRL (Baseline)",
    "SP_DQN":   "DQN",
    "SP_A2C":   "A2C",
    "SP_DDPG":  "DDPG",
    "BP_NoDRL": "No-DRL (Baseline)",
    "BP_DQN":   "DQN",
    "BP_A2C":   "A2C",
    "BP_DDPG":  "DDPG",
    # Banlic-Mamatid uses bare names
    "NoDRL":    "No-DRL (Baseline)",
    "DQN":      "DQN",
    "A2C":      "A2C",
    "DDPG":     "DDPG",
}

# Color palette: NoDRL always grey, DRL agents in distinct colours
AGENT_COLORS = {
    "No-DRL (Baseline)": "#888888",
    "DQN":               "#2196F3",
    "A2C":               "#4CAF50",
    "DDPG":              "#FF5722",
}

SCENARIO_RE = re.compile(r"(normal|slow|jam)", re.I)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_trips_xml(xml_path: Path) -> dict:
    """Return aggregated metrics from one *_trips_*.xml file."""
    veh_wait, veh_loss, veh_dur, veh_arrived = [], [], [], 0
    ped_wait, ped_loss = [], []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return {}

    for elem in root:
        tag = elem.tag.lower()
        a = elem.attrib

        if tag == "tripinfo":
            # Only count vehicles that actually arrived (not vaporized)
            if a.get("vaporized", "0") not in ("1", "True", "true"):
                veh_arrived += 1
            try: veh_wait.append(float(a["waitingTime"]))
            except: pass
            try: veh_loss.append(float(a["timeLoss"]))
            except: pass
            try: veh_dur.append(float(a["duration"]))
            except: pass

        elif tag == "personinfo":
            try: ped_wait.append(float(a["waitingTime"]))
            except: pass
            try: ped_loss.append(float(a["timeLoss"]))
            except: pass

    out = {}
    if veh_wait:   out["avg_waiting_time"]  = float(np.mean(veh_wait))
    if veh_loss:   out["avg_time_loss"]     = float(np.mean(veh_loss))
    if veh_dur:    out["avg_travel_time"]   = float(np.mean(veh_dur))
    if veh_arrived: out["throughput"]       = float(veh_arrived)
    if ped_wait:   out["ped_waiting_time"]  = float(np.mean(ped_wait))
    if ped_loss:   out["ped_time_loss"]     = float(np.mean(ped_loss))
    return out


def collect_location_data(batch_path: Path, prefix_filter: str = "ALL"):
    """
    Walk agent subfolders in batch_path, parse all *_trips_*.xml files.
    Returns data[agent][scenario][metric] = [run1_val, run2_val, ...]
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for agent_dir in sorted(batch_path.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name
        if prefix_filter != "ALL" and not agent.upper().startswith(prefix_filter.upper()):
            continue

        for xf in sorted(agent_dir.glob("*_trips_*.xml")):
            m = SCENARIO_RE.search(xf.name)
            if not m:
                continue
            scen = m.group(1).lower()
            row = parse_trips_xml(xf)
            for metric, val in row.items():
                data[agent][scen][metric].append(val)

    return data


# ── Stats ─────────────────────────────────────────────────────────────────────

def ci95(values):
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    se = np.nanstd(arr, ddof=1) / math.sqrt(n)
    t  = scipy_stats.t.ppf(0.975, df=n - 1)
    return float(t * se)


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_bar_chart(ax, scenario_data, agents, agent_label_map, metric_label,
                   higher_is_better, loc_name):
    """
    Draw a grouped bar chart on ax.
    scenario_data : {scen: {agent: [values]}}
    """
    scenarios  = [s for s in SCENARIO_ORDER if s in scenario_data]
    n_scenarios = len(scenarios)
    n_agents    = len(agents)
    bar_width   = 0.7 / n_agents
    x_base      = np.arange(n_scenarios)

    for ai, agent in enumerate(agents):
        label  = agent_label_map.get(agent, agent)
        color  = AGENT_COLORS.get(label, f"C{ai}")
        means  = []
        errors = []
        for scen in scenarios:
            vals = scenario_data.get(scen, {}).get(agent, [])
            if vals:
                means.append(float(np.mean(vals)))
                errors.append(ci95(vals))
            else:
                means.append(0.0)
                errors.append(0.0)

        xpos = x_base + (ai - n_agents / 2 + 0.5) * bar_width
        bars = ax.bar(xpos, means, bar_width * 0.9,
                      label=label, color=color, alpha=0.88, zorder=3)
        ax.errorbar(xpos, means, yerr=errors,
                    fmt='none', ecolor='black', elinewidth=1.2,
                    capsize=4, zorder=4)

        # value labels on top of bars
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(errors) * 0.1,
                        f"{val:.1f}", ha='center', va='bottom',
                        fontsize=7.5, color='#333333')

    ax.set_xticks(x_base)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_xlabel("Scenario", fontsize=10)

    direction = "(higher = better)" if higher_is_better else "(lower = better)"
    ax.set_title(f"{loc_name}\n{metric_label} {direction}", fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, title="Agent", title_fontsize=8,
              loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_for_location(loc_key, loc_name, data, outdir: Path, prefix_filter: str):
    outdir.mkdir(parents=True, exist_ok=True)

    agents = sorted(data.keys())
    if not agents:
        print(f"  No data found for {loc_key} with prefix={prefix_filter}")
        return

    # Build per-scenario lookup
    # scenario_data[scen][agent] = [values]
    for metric_key, metric_label, higher_is_better in METRICS:
        scenario_data = {}
        for scen in SCENARIO_ORDER:
            scenario_data[scen] = {}
            for agent in agents:
                vals = data[agent].get(scen, {}).get(metric_key, [])
                if vals:
                    scenario_data[scen][agent] = vals

        # skip if no data for this metric at all
        has_data = any(
            bool(vals)
            for scen in scenario_data.values()
            for vals in scen.values()
        )
        if not has_data:
            continue

        fig, ax = plt.subplots(figsize=(10, 5.5))
        make_bar_chart(ax, scenario_data, agents, AGENT_LABELS,
                       metric_label, higher_is_better, loc_name)
        plt.tight_layout()
        fname = f"{prefix_filter}_{metric_key}.png"
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)
        print(f"    saved: {fname}")

        # also save CSV summary
        rows = []
        for agent in agents:
            for scen in SCENARIO_ORDER:
                vals = scenario_data.get(scen, {}).get(agent, [])
                n    = len(vals)
                mean = float(np.mean(vals))  if n > 0 else None
                ci   = ci95(vals)            if n > 1 else None
                rows.append({
                    "location": loc_name, "agent": agent,
                    "agent_label": AGENT_LABELS.get(agent, agent),
                    "scenario": scen, "metric": metric_key,
                    "n": n, "mean": mean, "ci95": ci
                })
        pd.DataFrame(rows).to_csv(outdir / f"{prefix_filter}_{metric_key}.csv", index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",   default=".",
                        help="repo root (default: current directory)")
    parser.add_argument("--outdir", default="results_figs",
                        help="output directory")
    parser.add_argument("--prefix", default="SP",
                        choices=["SP", "BP", "ALL"],
                        help="which agent folders to include: SP (Signalized), "
                             "BP (Baseline Ped), or ALL")
    args = parser.parse_args()

    root    = Path(args.root)
    outroot = Path(args.outdir)

    for loc_key, loc_name in LOCATIONS:
        batch_path = root / loc_key / "batch_results"
        if not batch_path.exists():
            print(f"  Skipping {loc_key}: batch_results not found")
            continue

        print(f"\nProcessing {loc_name} ({loc_key})")
        data   = collect_location_data(batch_path, prefix_filter=args.prefix)
        outdir = outroot / loc_key
        plot_for_location(loc_key, loc_name, data, outdir, prefix_filter=args.prefix)

    print("\nAll done. Figures saved to:", outroot.resolve())


if __name__ == "__main__":
    main()
