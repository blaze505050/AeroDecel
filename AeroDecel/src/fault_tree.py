"""
src/fault_tree.py — Fault Tree Analysis (FTA) for EDL Mission
==============================================================
Builds a probabilistic fault tree for the EDL sequence and computes
mission success probability from component reliabilities.

Fault tree structure (AND/OR gates):
  Mission Loss (Top event)
  ├── OR gate: any major failure
  │   ├── AND gate: Entry failure
  │   │   ├── TPS burnthrough (basic event)
  │   │   └── Structural overload during peak g (basic event)
  │   ├── OR gate: Descent failure
  │   │   ├── Chute fails to deploy (basic event)
  │   │   ├── Chute structural failure (basic event)
  │   │   └── Aeroelastic flutter damage (basic event)
  │   └── OR gate: Landing failure
  │       ├── Excessive landing velocity (basic event)
  │       ├── Terrain hazard impact (basic event)
  │       └── Pendulum oscillation tip-over (basic event)

Each basic event has:
  • Nominal probability (design point)
  • 3-sigma uncertainty (log-normal)
  • Sensitivity to design variables

Output:
  • Mission success probability P(success)
  • Minimal cut sets (which combinations cause failure)
  • Importance measures: Birnbaum, Fussell-Vesely
  • Sensitivity tornado chart
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# FAULT TREE NODES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BasicEvent:
    """Leaf node of the fault tree — a single failure mode."""
    name:           str
    description:    str
    prob_nominal:   float          # P(failure) at nominal conditions
    prob_sigma:     float          # log-normal sigma for uncertainty
    category:       str            # "TPS" | "Structural" | "Chute" | "Landing"
    mitigation:     str = ""       # how it's mitigated in design

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Sample failure probability from log-normal uncertainty."""
        log_mu    = np.log(max(self.prob_nominal, 1e-12))
        log_sigma = self.prob_sigma
        samples   = rng.lognormal(log_mu, log_sigma, n)
        return np.clip(samples, 0.0, 1.0)


@dataclass
class Gate:
    """Intermediate gate node (AND or OR)."""
    name:     str
    gate_type: Literal["AND", "OR"]
    inputs:   list        # list of Gate or BasicEvent
    description: str = ""

    def probability(self) -> float:
        """
        Compute gate failure probability from input probabilities.
        AND gate: P(all fail) = Π P_i
        OR  gate: P(any fail) = 1 - Π(1 - P_i)
        """
        probs = [inp.probability() if isinstance(inp, Gate)
                 else inp.prob_nominal for inp in self.inputs]
        if self.gate_type == "AND":
            return float(np.prod(probs))
        else:  # OR
            return float(1 - np.prod([1-p for p in probs]))

    def probability_mc(self, samples_dict: dict) -> float:
        """Monte Carlo probability from sampled basic event probabilities."""
        probs = []
        for inp in self.inputs:
            if isinstance(inp, Gate):
                probs.append(inp.probability_mc(samples_dict))
            else:
                probs.append(float(samples_dict.get(inp.name, inp.prob_nominal)))
        if self.gate_type == "AND":
            return float(np.prod(probs))
        else:
            return float(1 - np.prod([1-p for p in probs]))


# ══════════════════════════════════════════════════════════════════════════════
# EDL FAULT TREE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_edl_fault_tree(
    p_tps_burnthrough:  float = 0.02,
    p_struct_overload:  float = 0.005,
    p_chute_no_deploy:  float = 0.008,
    p_chute_structural: float = 0.003,
    p_flutter_damage:   float = 0.004,
    p_hard_landing:     float = 0.015,
    p_terrain_hazard:   float = 0.025,
    p_tipover:          float = 0.006,
    sf_tps:             float = 1.5,
    sf_structure:       float = 2.0,
) -> tuple[Gate, list[BasicEvent]]:
    """
    Build the EDL fault tree with given component reliabilities.

    SF-based probability scaling:
      P(failure) ∝ 1/SF²   (simplified structural reliability model)
    """
    # Scale probabilities by safety factors
    p_tps  = p_tps_burnthrough * min(1, 1/max(sf_tps, 0.1)**1.5)
    p_stru = p_struct_overload  * min(1, 1/max(sf_structure, 0.1)**2)

    # ── Basic Events ──────────────────────────────────────────────────────────
    E = {
        "tps_burnthrough": BasicEvent(
            "TPS Burnthrough", "TPS ablates through before chute deploy",
            p_tps, 0.8, "TPS",
            "Margin: SF_TPS > 1.5, 3-sigma recession check"),
        "struct_overload": BasicEvent(
            "Structural Overload", "Peak g-load exceeds structural limit",
            p_stru, 0.6, "Structural",
            "Entry angle constrained to limit g-load < 20g"),
        "chute_no_deploy": BasicEvent(
            "Chute Fails to Deploy", "Mortar misfire or bag retention failure",
            p_chute_no_deploy, 0.7, "Chute",
            "Redundant mortar, dynamic pressure trigger"),
        "chute_structural": BasicEvent(
            "Chute Structural Failure", "Gore tear or riser failure at inflation",
            p_chute_structural, 0.65, "Chute",
            "Opening shock SF > 3, certified materials"),
        "flutter_damage": BasicEvent(
            "Aeroelastic Flutter Damage", "Canopy flutter causes gore fatigue failure",
            p_flutter_damage, 0.9, "Chute",
            "Flutter analysis: v_critical > max deployment speed"),
        "hard_landing": BasicEvent(
            "Excessive Landing Velocity", "Terminal velocity exceeds survivable limit",
            p_hard_landing, 0.7, "Landing",
            "Parachute sized for v_land < 15 m/s, 3-sigma"),
        "terrain_hazard": BasicEvent(
            "Terrain Hazard Impact", "Landing ellipse overlaps with slopes > 15°",
            p_terrain_hazard, 1.0, "Landing",
            "Landing ellipse < 10 km CEP, hazard avoidance"),
        "tipover": BasicEvent(
            "Pendulum Oscillation Tip-Over", "Payload tips on contact due to swing",
            p_tipover, 0.8, "Landing",
            "Pendulum damping via riser geometry"),
    }

    # ── Gate structure ─────────────────────────────────────────────────────────
    entry_gate = Gate(
        "Entry Failure", "AND",
        [E["tps_burnthrough"], E["struct_overload"]],
        "Both TPS and structural failure must occur simultaneously"
    )
    descent_gate = Gate(
        "Descent Failure", "OR",
        [E["chute_no_deploy"], E["chute_structural"], E["flutter_damage"]],
        "Any chute failure causes descent loss"
    )
    landing_gate = Gate(
        "Landing Failure", "OR",
        [E["hard_landing"], E["terrain_hazard"], E["tipover"]],
        "Any landing event"
    )
    top_gate = Gate(
        "Mission Loss", "OR",
        [entry_gate, descent_gate, landing_gate],
        "Any major failure leads to mission loss"
    )

    return top_gate, list(E.values())


# ══════════════════════════════════════════════════════════════════════════════
# FAULT TREE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class FaultTreeAnalysis:
    """
    Computes mission reliability metrics from a fault tree.

    Methods
    -------
    - Nominal probability (analytical)
    - Monte Carlo uncertainty propagation
    - Birnbaum importance measures
    - Fussell-Vesely importance
    - Minimal cut sets (top 3 by probability)
    """

    def __init__(self, top_gate: Gate, basic_events: list[BasicEvent]):
        self.top    = top_gate
        self.events = {e.name: e for e in basic_events}

    def mission_success_probability(self) -> float:
        """P(success) = 1 - P(top event)."""
        return 1.0 - self.top.probability()

    def monte_carlo(self, n_samples: int = 10_000,
                    seed: int = 0) -> dict:
        """
        Monte Carlo uncertainty on mission success probability.
        Propagates log-normal uncertainty in each basic event.
        """
        rng = np.random.default_rng(seed)
        p_success_samples = np.zeros(n_samples)

        for s in range(n_samples):
            # Sample each basic event probability
            sampled = {}
            for name, event in self.events.items():
                sampled[name] = float(event.sample(rng, 1)[0])

            # Compute top event probability with sampled values
            p_top = self.top.probability_mc(sampled)
            p_success_samples[s] = 1.0 - p_top

        return {
            "p_success_mean":    float(np.mean(p_success_samples)),
            "p_success_std":     float(np.std(p_success_samples)),
            "p_success_p05":     float(np.percentile(p_success_samples, 5)),
            "p_success_p50":     float(np.percentile(p_success_samples, 50)),
            "p_success_p95":     float(np.percentile(p_success_samples, 95)),
            "p_success_samples": p_success_samples,
            "n_samples":         n_samples,
        }

    def birnbaum_importance(self) -> dict:
        """
        Birnbaum structural importance: I_B(i) = P(top | e_i=1) - P(top | e_i=0).
        Measures how much each event contributes to top event probability.
        """
        importance = {}
        for name, event in self.events.items():
            # P(top | e_i = 1)
            orig = event.prob_nominal
            event.prob_nominal = 1.0
            p1 = self.top.probability()
            # P(top | e_i = 0)
            event.prob_nominal = 0.0
            p0 = self.top.probability()
            event.prob_nominal = orig
            importance[name] = round(float(p1 - p0), 8)

        return dict(sorted(importance.items(), key=lambda x: -abs(x[1])))

    def fussell_vesely_importance(self) -> dict:
        """
        Fussell-Vesely importance: I_FV(i) = P(top via e_i) / P(top).
        Fraction of top event probability contributed by each event.
        """
        p_top = max(self.top.probability(), 1e-15)
        importance = {}
        for name, event in self.events.items():
            orig = event.prob_nominal
            event.prob_nominal = 0.0
            p_top_excl = self.top.probability()
            event.prob_nominal = orig
            importance[name] = round(float((p_top - p_top_excl) / p_top), 6)

        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def minimal_cut_sets(self) -> list[list[str]]:
        """
        Enumerate minimal cut sets (top-3 most probable combinations).
        A cut set is a minimal set of basic events whose simultaneous failure
        causes the top event.
        """
        events_list = list(self.events.keys())
        n = len(events_list)
        mcs = []

        # Order-1 cut sets
        for e in events_list:
            orig = {k: v.prob_nominal for k, v in self.events.items()}
            for k in events_list:
                self.events[k].prob_nominal = 1.0 if k == e else 0.0
            if self.top.probability() > 0.5:
                mcs.append(([e], orig[e]))
            for k, v in orig.items():
                self.events[k].prob_nominal = v

        # Order-2 cut sets
        for i in range(n):
            for j in range(i+1, n):
                e1, e2 = events_list[i], events_list[j]
                orig = {k: v.prob_nominal for k, v in self.events.items()}
                for k in events_list:
                    self.events[k].prob_nominal = 1.0 if k in (e1, e2) else 0.0
                p_set = orig[e1] * orig[e2]
                if self.top.probability() > 0.5:
                    mcs.append(([e1, e2], p_set))
                for k, v in orig.items():
                    self.events[k].prob_nominal = v

        mcs.sort(key=lambda x: -x[1])
        return [m[0] for m in mcs[:10]]

    def full_report(self, n_mc: int = 5_000) -> dict:
        """Generate complete FTA report."""
        p_success = self.mission_success_probability()
        mc        = self.monte_carlo(n_samples=n_mc)
        birnbaum  = self.birnbaum_importance()
        fv        = self.fussell_vesely_importance()
        mcs       = self.minimal_cut_sets()

        return {
            "P_mission_success":      round(p_success, 8),
            "P_mission_failure":      round(1 - p_success, 8),
            "P_success_p05_MC":       mc["p_success_p05"],
            "P_success_p50_MC":       mc["p_success_p50"],
            "P_success_p95_MC":       mc["p_success_p95"],
            "birnbaum_importance":    birnbaum,
            "fussell_vesely":         fv,
            "minimal_cut_sets":       mcs,
            "mc_result":              mc,
            "basic_events": {
                name: {"prob": e.prob_nominal, "category": e.category,
                       "description": e.description}
                for name, e in self.events.items()
            },
        }


def plot_fault_tree(report: dict, save_path: str = "outputs/fault_tree.png"):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import gaussian_kde

    matplotlib.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })
    TX="#c8d8f0"; C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"; CR="#ff4560"

    fig = plt.figure(figsize=(22, 12), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.05, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    p_success = report["P_mission_success"]
    p_fail    = report["P_mission_failure"]
    col_main  = C3 if p_success > 0.99 else (C4 if p_success > 0.95 else CR)

    # Main success probability
    a = gax(0, 0); a.axis("off")
    a.text(0.5, 0.75, f"P(SUCCESS)", ha="center", transform=a.transAxes,
           fontsize=12, fontweight="bold", color="#556688")
    a.text(0.5, 0.50, f"{p_success:.6f}", ha="center", transform=a.transAxes,
           fontsize=22, fontweight="bold", color=col_main)
    a.text(0.5, 0.30, f"P(FAILURE) = {p_fail:.2e}", ha="center",
           transform=a.transAxes, fontsize=10, color=CR)
    a.text(0.5, 0.12, f"MC p05={report['P_success_p05_MC']:.5f}\n"
           f"MC p50={report['P_success_p50_MC']:.5f}\n"
           f"MC p95={report['P_success_p95_MC']:.5f}",
           ha="center", transform=a.transAxes, fontsize=9, color=TX)

    # MC distribution
    a = gax(0, 1)
    mc_samples = report["mc_result"]["p_success_samples"]
    a.hist(mc_samples, bins=50, color=col_main, alpha=0.65, density=True, edgecolor="none")
    try:
        kde = gaussian_kde(mc_samples)
        xs  = np.linspace(mc_samples.min(), mc_samples.max(), 200)
        a.plot(xs, kde(xs), color=C1, lw=2)
    except: pass
    a.axvline(p_success, color=C4, lw=1.5, ls="--", label=f"Nominal={p_success:.5f}")
    a.axvline(report["P_success_p05_MC"], color=CR, lw=1, ls=":", label="5th pct")
    a.legend(fontsize=7.5)
    a.set_title("P(success) Uncertainty (MC)", fontweight="bold")
    a.set_xlabel("P(success)"); a.set_ylabel("Density")

    # Birnbaum importance
    a = gax(0, 2)
    bm = report["birnbaum_importance"]
    names_bm = list(bm.keys())[:8]; vals_bm = [bm[k] for k in names_bm]
    short_names = [n.replace("_", "\n") for n in names_bm]
    colors_bm   = [CR if v > 0.01 else (C4 if v > 0.001 else C3) for v in vals_bm]
    a.barh(short_names[::-1], vals_bm[::-1], color=colors_bm[::-1], alpha=0.75)
    a.set_title("Birnbaum Importance I_B(i)", fontweight="bold")
    a.set_xlabel("I_B = P(top|e_i=1) − P(top|e_i=0)")

    # Fussell-Vesely importance
    a = gax(0, 3)
    fv = report["fussell_vesely"]
    names_fv = list(fv.keys())[:8]; vals_fv = [fv[k]*100 for k in names_fv]
    short_fv  = [n.replace("_", "\n") for n in names_fv]
    colors_fv = [CR if v > 10 else (C4 if v > 2 else C3) for v in vals_fv]
    a.barh(short_fv[::-1], vals_fv[::-1], color=colors_fv[::-1], alpha=0.75)
    a.set_title("Fussell-Vesely Importance I_FV", fontweight="bold")
    a.set_xlabel("I_FV [%]")

    # Basic event probabilities
    a = gax(1, 0)
    be = report["basic_events"]
    cat_colors = {"TPS": CR, "Structural": C2, "Chute": C4, "Landing": C1}
    names_be = list(be.keys()); probs_be = [be[k]["prob"] for k in names_be]
    cats_be  = [be[k]["category"] for k in names_be]
    colors_be= [cat_colors.get(c, "#888") for c in cats_be]
    bars = a.barh([n.replace("_","\n") for n in names_be],
                   probs_be, color=colors_be, alpha=0.75)
    a.set_xscale("log"); a.set_title("Basic Event Probabilities", fontweight="bold")
    a.set_xlabel("P(failure)")
    from matplotlib.patches import Patch
    legend_el = [Patch(color=v, label=k) for k, v in cat_colors.items()]
    a.legend(handles=legend_el, fontsize=7)

    # Category breakdown pie
    a = gax(1, 1); a.axis("off")
    cats = {}
    for name, ev in be.items():
        cats[ev["category"]] = cats.get(ev["category"], 0) + ev["prob"]
    wedge_col = [cat_colors.get(k, "#888") for k in cats]
    a.pie(list(cats.values()), labels=list(cats.keys()),
          colors=wedge_col, autopct="%1.1f%%", startangle=90)
    a.set_title("Failure probability by category", fontweight="bold")

    # Minimal cut sets table
    a = gax(1, 2); a.axis("off")
    a.text(0.5, 0.98, "MINIMAL CUT SETS", ha="center", transform=a.transAxes,
           fontsize=10, fontweight="bold", color=C1)
    mcs = report["minimal_cut_sets"]
    for j, cs in enumerate(mcs[:6]):
        y = 0.86 - j*0.14
        cs_str = " ∧ ".join([c.replace("_", " ") for c in cs])
        a.text(0.05, y, f"{j+1}.", transform=a.transAxes, fontsize=8.5, color="#556688")
        a.text(0.12, y, cs_str, transform=a.transAxes, fontsize=8, color=TX,
               wrap=True)

    # Fault tree diagram (simplified text)
    a = gax(1, 3); a.axis("off")
    a.text(0.5, 0.98, "FAULT TREE STRUCTURE", ha="center", transform=a.transAxes,
           fontsize=10, fontweight="bold", color=C1)
    tree_txt = (
        "Top: Mission Loss\n"
        "  └─ OR\n"
        "      ├─ Entry Fail (AND)\n"
        "      │   ├─ TPS Burnthrough\n"
        "      │   └─ Struct Overload\n"
        "      ├─ Descent Fail (OR)\n"
        "      │   ├─ Chute No Deploy\n"
        "      │   ├─ Chute Struct Fail\n"
        "      │   └─ Flutter Damage\n"
        "      └─ Landing Fail (OR)\n"
        "          ├─ Hard Landing\n"
        "          ├─ Terrain Hazard\n"
        "          └─ Tip-Over"
    )
    a.text(0.02, 0.88, tree_txt, transform=a.transAxes, fontsize=8,
           color=TX, fontfamily="monospace", va="top")

    fig.text(0.5, 0.955,
             f"EDL Fault Tree Analysis  |  "
             f"P(success)={p_success:.6f}  |  "
             f"P(failure)={p_fail:.2e}  |  "
             f"n_MC={report['mc_result']['n_samples']:,}",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Fault tree plot saved: {save_path}")
    plt.close(fig)


def run(sf_tps: float = 1.5, sf_structure: float = 2.0,
        n_mc: int = 5_000, verbose: bool = True) -> dict:
    """Run full fault tree analysis."""
    import matplotlib; matplotlib.use("Agg")

    top, events = build_edl_fault_tree(sf_tps=sf_tps, sf_structure=sf_structure)
    fta = FaultTreeAnalysis(top, events)
    report = fta.full_report(n_mc=n_mc)

    if verbose:
        print(f"\n[FTA] Mission success probability: {report['P_mission_success']:.6f}")
        print(f"      Mission failure probability:  {report['P_mission_failure']:.2e}")
        print(f"      MC p05/p50/p95: {report['P_success_p05_MC']:.5f} / "
              f"{report['P_success_p50_MC']:.5f} / {report['P_success_p95_MC']:.5f}")
        print(f"\n  Top importance (Birnbaum):")
        for name, val in list(report["birnbaum_importance"].items())[:4]:
            print(f"    {name:30s}: {val:.6f}")
        print(f"\n  Minimal cut sets (top 3):")
        for cs in report["minimal_cut_sets"][:3]:
            print(f"    {' ∧ '.join(cs)}")

    plot_fault_tree(report)
    return report
