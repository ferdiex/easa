"""
aggregate_sweep_results.py

Aggregates all easa_log_*.csv / *.json pairs in the current directory
(produced by run_dopamine_sweep.sh) into:
  - sweep_summary.csv: one row per run, with dopamine, successful-deposit
    count, episode length, and a count of "gate-blend" events (wheels
    moving while an arm command is issued in the same decision step --
    the proxy for the SWITCHING_CONSTANT motor-gating mechanism found
    this session, see notas_sesion_bg90s_real_v2.md).
  - sweep_summary_by_dopamine.csv: the same, aggregated (mean/std/median/
    failure rate) per dopamine value across repetitions.
  - sweep_statistics.csv: NEW -- significance testing for the paper.
    Every non-baseline dopamine value is compared against BASELINE_DOPAMINE
    (default 0.2, the calibrated point) two ways:
      * Fisher's exact test on the 2x2 table of (failed run / completed
        at least one deposit), since total_failure_rate is a proportion
        and Fisher's exact is the right test for two independent binomial
        samples (more accurate than chi-square at these sample sizes,
        exact rather than asymptotic).
      * Mann-Whitney U on deposits_per_1000_steps, since that
        distribution is confirmed non-normal (heavy-tailed / bimodal --
        see the session notes), so a t-test's normality assumption does
        not hold; Mann-Whitney does not require it.
    A Kruskal-Wallis test across all dopamine groups is also reported
    (the omnibus non-parametric equivalent of one-way ANOVA), to support
    the claim that dopamine has *some* effect before looking at which
    pairs differ.
    All pairwise p-values are Holm-Bonferroni corrected across the set of
    comparisons run (4 comparisons against the baseline, for each of the
    two tests) -- reported as both raw and corrected p-values, since
    testing multiple dopamine values against the same baseline inflates
    the false-positive rate if left uncorrected.
  - sweep_summary.png: three panels -- the original two means (kept for
    transparency, explicitly flagged as noisy) plus the failure-rate
    panel with Wilson score confidence intervals (the correct interval
    for a binomial proportion -- a plain normal-approximation interval
    can extend below 0% or above 100%, which Wilson avoids) and
    significance stars for pairs that survive the Holm-Bonferroni
    correction against the baseline.

Usage:
    python3 aggregate_sweep_results.py [directory] [--baseline 0.2]
(directory defaults to the current directory; baseline defaults to 0.2,
 the calibrated non-negotiable dopamine value for this model)
"""
import sys
import os
import glob
import json
import csv
import argparse
from collections import defaultdict

from scipy import stats

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def count_successful_deposits(csv_path):
    """A deposit counts as successful when the active channel leaves
    corner_deposit while hunger has dropped below 0.5 -- same heuristic
    used throughout this session's manual log analysis (a real release
    resets hunger to fear/3, which is well under 0.5 in practice)."""
    deposits = 0
    prev_channel = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            channel = row["active_channel"]
            if channel != prev_channel:
                if prev_channel == "corner_deposit" and float(row["hunger"]) < 0.5:
                    deposits += 1
                prev_channel = channel
    return deposits


def count_gate_blend_events(csv_path):
    """A 'walk+arm' gate-blend event: wheels clearly moving
    (|left_speed_cmd| or |right_speed_cmd| > 1) in the same decision step
    an arm_target_cmd is issued. This is the observable signature of the
    SWITCHING_CONSTANT motor-gating mechanism (see the 'La pregunta
    huntingtoniana' section of notas_sesion_bg90s_real_v2.md) -- NOT
    simultaneous thalamus crossing, which this session confirmed does not
    happen at these dopamine levels."""
    events = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            left = float(row["left_speed_cmd"])
            right = float(row["right_speed_cmd"])
            walking = abs(left) > 1 or abs(right) > 1
            if walking and row["arm_target_cmd"] != "":
                events += 1
    return events


def episode_length_steps(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]["step"]) if rows else 0


def wilson_score_interval(successes, n, confidence=0.95):
    """95% Wilson score confidence interval for a binomial proportion.
    Used instead of the plain normal-approximation interval (p +/- 1.96 *
    sqrt(p(1-p)/n)) because that one can extend below 0% or above 100%
    when p is close to an extreme (exactly what happens here: the
    baseline dopamine has only 1/20 failures, a proportion near 0). Wilson
    stays within [0, 1] by construction and is the standard recommended
    interval for this situation."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction. Returns corrected p-values in
    the SAME order as the input. Used instead of a single flat Bonferroni
    correction because Holm is uniformly more powerful (rejects at least
    as many true effects) while controlling the same family-wise error
    rate -- there is no statistical reason to prefer plain Bonferroni here."""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (n - rank) * p_values[idx]
        running_max = max(running_max, adjusted)
        corrected[idx] = min(1.0, running_max)
    return corrected


def write_markdown_table(path, caption, headers, data_rows):
    """Writes a simple, clean markdown table (caption + header + rows).
    Markdown tables paste cleanly into most LaTeX pipelines (via pandoc)
    and into Word -- this is deliberately plain (no styling) so it is
    easy to convert to whatever the target journal's table format needs,
    rather than trying to guess that format here."""
    with open(path, "w") as f:
        f.write(f"{caption}\n\n")
        f.write("| " + " | ".join(str(h) for h in headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in data_rows:
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("--baseline", type=float, default=0.2,
                         help="dopamine value to compare every other value against "
                              "(default 0.2, the calibrated point)")
    parser.add_argument("--include", choices=["fixed", "converge", "both"], default="fixed",
                         help="NEW: which runs to include, by CONVERGE_MODE in their "
                              "metadata. Defaults to 'fixed' (CONVERGE_MODE=False, the "
                              "normal MAXTIME-based runs) ONLY -- this is deliberately "
                              "not 'both', because mixing fixed-mode and convergence-mode "
                              "runs into the same dopamine bucket silently contaminates "
                              "the statistics (this happened once this session: a 6-run "
                              "convergence pilot got folded into a 165-run fixed-mode "
                              "sweep without anyone noticing until the run count looked "
                              "off). Pass --include converge to analyze ONLY the "
                              "convergence-mode runs, or --include both if you genuinely "
                              "want them pooled (rarely what you want for a comparison).")
    args = parser.parse_args()
    directory = args.directory
    baseline_dopamine = args.baseline

    # NEW, to avoid --include fixed and --include converge overwriting
    # each other's output files: every output filename gets a suffix
    # reflecting which runs went into it. Empty for the default (fixed)
    # so existing usage/filenames are unaffected; non-empty otherwise.
    suffix = {"fixed": "", "converge": "_converge", "both": "_both"}[args.include]

    def out(filename):
        """Inserts `suffix` before the file extension, e.g.
        out("sweep_summary.csv") -> "sweep_summary_converge.csv" when
        --include converge, or the unchanged name for the default."""
        stem, ext = os.path.splitext(filename)
        return os.path.join(directory, f"{stem}{suffix}{ext}")

    json_paths = sorted(glob.glob(os.path.join(directory, "easa_log_*.json")))
    if not json_paths:
        print(f"No easa_log_*.json metadata files found in {directory}. "
              f"Run run_dopamine_sweep.sh first.")
        sys.exit(1)

    rows = []
    n_skipped_by_mode = 0
    for json_path in json_paths:
        with open(json_path) as f:
            meta = json.load(f)
        is_converge = bool(meta.get("CONVERGE_MODE", False))
        if args.include == "fixed" and is_converge:
            n_skipped_by_mode += 1
            continue
        if args.include == "converge" and not is_converge:
            n_skipped_by_mode += 1
            continue
        csv_path = os.path.join(directory, meta["log_path"])
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found (metadata without matching log)")
            continue

        deposits = count_successful_deposits(csv_path)
        gate_events = count_gate_blend_events(csv_path)
        length = episode_length_steps(csv_path)
        rows.append({
            "run_tag": meta.get("run_tag", ""),
            "dopamine_d1": meta["dopamine_d1"],
            "dopamine_d2": meta["dopamine_d2"],
            "MAXTIME": meta["MAXTIME"],
            "converge_mode": is_converge,
            "episode_length_steps": length,
            "successful_deposits": deposits,
            "deposits_per_1000_steps": 1000 * deposits / length if length else 0.0,
            "gate_blend_events": gate_events,
            "gate_blend_events_per_1000_steps": 1000 * gate_events / length if length else 0.0,
            "csv_path": meta["log_path"],
        })

    # Per-run table.
    if not rows:
        print(f"No runs matched --include {args.include} "
              f"({n_skipped_by_mode} skipped by that filter). Nothing to aggregate.")
        sys.exit(1)
    with open(out("sweep_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {len(rows)} runs (--include {args.include}, "
          f"{n_skipped_by_mode} skipped by that filter) -> sweep_summary.csv")

    # Aggregate by dopamine value (mean/std across repetitions).
    by_dopamine = defaultdict(list)
    for r in rows:
        by_dopamine[r["dopamine_d1"]].append(r)

    agg_rows = []
    for dopamine in sorted(by_dopamine):
        group = by_dopamine[dopamine]
        deposits = np.array([g["deposits_per_1000_steps"] for g in group])
        gate_events = np.array([g["gate_blend_events_per_1000_steps"] for g in group])
        # NEW, at the user's request: mean/std are misleading here -- the
        # per-run distribution of successful_deposits is heavy-tailed /
        # bimodal at dopamine values far from the calibrated default
        # (a few runs do very well, many get exactly zero), which the
        # mean+-std error bars misrepresent (e.g. an error bar that dips
        # below zero, which is not physically meaningful for a count).
        # The fraction of runs with zero successful deposits -- complete
        # task failure within the episode -- is a more robust summary of
        # the same effect, and turned out to tell a cleaner story: the
        # calibrated default (0.2) is close to the failure-rate minimum,
        # not just "a reasonable point on a noisy mean curve".
        raw_deposits = np.array([g["successful_deposits"] for g in group])
        total_failure_rate = float((raw_deposits == 0).mean())
        n_failures = int((raw_deposits == 0).sum())
        ci_lower, ci_upper = wilson_score_interval(n_failures, len(group))
        agg_rows.append({
            "dopamine": dopamine,
            "n_runs": len(group),
            "deposits_per_1000_steps_mean": deposits.mean(),
            "deposits_per_1000_steps_std": deposits.std(),
            "deposits_per_1000_steps_median": float(np.median(deposits)),
            "total_failure_rate": total_failure_rate,  # fraction of runs with 0 deposits
            "total_failure_rate_wilson_ci_lower": ci_lower,
            "total_failure_rate_wilson_ci_upper": ci_upper,
            "gate_blend_events_per_1000_steps_mean": gate_events.mean(),
            "gate_blend_events_per_1000_steps_std": gate_events.std(),
        })

    with open(out("sweep_summary_by_dopamine.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"[OK] {len(agg_rows)} dopamine values -> sweep_summary_by_dopamine.csv")

    # NEW: significance testing against the calibrated baseline.
    if baseline_dopamine not in by_dopamine:
        print(f"[WARN] baseline dopamine {baseline_dopamine} not found in the "
              f"data, skipping significance testing")
        stat_rows = []
    else:
        baseline_group = by_dopamine[baseline_dopamine]
        baseline_deposits = np.array([g["deposits_per_1000_steps"] for g in baseline_group])
        baseline_raw = np.array([g["successful_deposits"] for g in baseline_group])
        baseline_failures = int((baseline_raw == 0).sum())
        baseline_n = len(baseline_group)

        comparison_dopamines = [d for d in sorted(by_dopamine) if d != baseline_dopamine]
        fisher_p = []
        mwu_p = []
        stat_rows_partial = []
        for dopamine in comparison_dopamines:
            group = by_dopamine[dopamine]
            group_deposits = np.array([g["deposits_per_1000_steps"] for g in group])
            group_raw = np.array([g["successful_deposits"] for g in group])
            group_failures = int((group_raw == 0).sum())
            group_n = len(group)

            # Fisher's exact test on the 2x2 table [[failures, successes],
            # [failures, successes]] for baseline vs. this dopamine value.
            table = [[baseline_failures, baseline_n - baseline_failures],
                     [group_failures, group_n - group_failures]]
            odds_ratio, p_fisher = stats.fisher_exact(table)
            fisher_p.append(p_fisher)

            # Mann-Whitney U on the continuous deposits-per-1000-steps
            # metric, non-parametric because that distribution is
            # confirmed non-normal (heavy-tailed/bimodal, see module
            # docstring).
            _, p_mwu = stats.mannwhitneyu(baseline_deposits, group_deposits, alternative="two-sided")
            mwu_p.append(p_mwu)

            stat_rows_partial.append({
                "dopamine": dopamine,
                "baseline_dopamine": baseline_dopamine,
                "fisher_odds_ratio": odds_ratio,
                "fisher_p_raw": p_fisher,
                "mannwhitney_p_raw": p_mwu,
            })

        fisher_p_corrected = holm_bonferroni(np.array(fisher_p))
        mwu_p_corrected = holm_bonferroni(np.array(mwu_p))
        stat_rows = []
        for i, r in enumerate(stat_rows_partial):
            r["fisher_p_holm_corrected"] = float(fisher_p_corrected[i])
            r["mannwhitney_p_holm_corrected"] = float(mwu_p_corrected[i])
            r["significant_at_0.05"] = bool(fisher_p_corrected[i] < 0.05)
            stat_rows.append(r)

        # Omnibus Kruskal-Wallis across ALL dopamine groups (not just vs.
        # baseline) -- the non-parametric one-way-ANOVA-equivalent, run
        # first in the usual reporting order to justify doing the pairwise
        # tests above at all (only meaningful if the omnibus test itself
        # is significant).
        all_deposit_groups = [np.array([g["deposits_per_1000_steps"] for g in by_dopamine[d]])
                               for d in sorted(by_dopamine)]
        kw_stat, kw_p = stats.kruskal(*all_deposit_groups)
        print(f"[OK] Kruskal-Wallis across all {len(all_deposit_groups)} dopamine groups: "
              f"H={kw_stat:.3f}, p={kw_p:.4g}")

        with open(out("sweep_statistics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stat_rows)
        print(f"[OK] {len(stat_rows)} pairwise comparisons vs. baseline={baseline_dopamine} "
              f"-> sweep_statistics.csv (Holm-Bonferroni corrected)")

    # NEW: individual, separately-numbered figures instead of one combined
    # multi-panel PNG -- a wide 3-panel figure often doesn't fit a
    # two-column journal layout, and journals generally want each figure
    # as its own file anyway (Figure 1, Figure 2, ... in the manuscript,
    # each with its own file for submission).
    # NOTE: the original Figure 1 (task performance, mean +/- std) was
    # dropped from the manuscript after review -- the failure-rate
    # figure (below) is the metric actually used, and keeping a figure
    # the text itself says not to trust read badly to a reviewer. Not
    # regenerated here anymore.
    dopamines = [r["dopamine"] for r in agg_rows]

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.errorbar(dopamines,
                 [r["gate_blend_events_per_1000_steps_mean"] for r in agg_rows],
                 yerr=[r["gate_blend_events_per_1000_steps_std"] for r in agg_rows],
                 marker="o", color="tab:red", capsize=3)
    ax2.set_xlabel("dopamine (D1 = D2)")
    ax2.set_ylabel("gate-blend events / 1000 steps")
    ax2.set_title("Motor-gate blending vs. dopamine\n(the 'walk+arm' mechanism)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2_path = out("figure4_gate_blending.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Figure 4 -> {fig2_path}")

    # Wilson CI error bars (asymmetric, from agg_rows) instead of a bare
    # bar, plus a significance star for any value found significantly
    # different from the baseline after Holm-Bonferroni correction.
    failure_pct = [100 * r["total_failure_rate"] for r in agg_rows]
    # max(0.0, ...): the Wilson bounds are clamped to [0, 1] inside
    # wilson_score_interval(), but total_failure_rate and the returned
    # bound are computed via two independent floating-point paths, so for
    # some n/k combinations the subtraction below lands a few ULPs below
    # zero (e.g. -2e-16) -- mathematically zero, but matplotlib's yerr
    # rejects any negative value, however tiny.
    ci_lower = [max(0.0, 100 * (r["total_failure_rate"] - r["total_failure_rate_wilson_ci_lower"])) for r in agg_rows]
    ci_upper = [max(0.0, 100 * (r["total_failure_rate_wilson_ci_upper"] - r["total_failure_rate"])) for r in agg_rows]
    significant = {r["dopamine"]: r["significant_at_0.05"] for r in stat_rows} if stat_rows else {}

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    bars = ax3.bar([str(d) for d in dopamines], failure_pct, color="tab:orange",
                   yerr=[ci_lower, ci_upper], capsize=4, ecolor="black")
    # NEW: extra headroom above the tallest bar+CI+star, so the star
    # never collides with the two-line title above it (this happened
    # when a bar's CI reached close to 100%, e.g. the convergence-mode
    # D=1.0 bar).
    tallest_top = max(f + u for f, u in zip(failure_pct, ci_upper))
    ax3.set_ylim(0, tallest_top * 1.25)
    for bar, dopamine in zip(bars, dopamines):
        if significant.get(dopamine, False):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ci_upper) * 0.15,
                      "*", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax3.set_xlabel("dopamine (D1 = D2)")
    ax3.set_ylabel("% of runs with zero successful deposits")
    ax3.set_title(f"Total task failure rate vs. dopamine\n"
                  f"(95% Wilson CI; * = p<0.05 vs. D={baseline_dopamine}, Holm-Bonferroni corrected)")
    ax3.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig3_path = out("figure3_failure_rate.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[OK] Figure 3 -> {fig3_path}")

    # NEW: publication-ready tables (markdown -- copy/pastes cleanly into
    # LaTeX via pandoc, or into Word) alongside the existing CSVs, since
    # journals generally want both figures AND tables, not just raw data.
    write_markdown_table(
        out("table2_summary_by_dopamine.md"),
        "Table 2. Task performance and motor-gate blending by dopamine level.",
        ["Dopamine", "n", "Deposits/1000 steps (mean +/- std)",
         "Failure rate % (95% CI)", "Gate-blend events/1000 steps (mean +/- std)"],
        [[r["dopamine"], r["n_runs"],
          f"{r['deposits_per_1000_steps_mean']:.3f} +/- {r['deposits_per_1000_steps_std']:.3f}",
          f"{100*r['total_failure_rate']:.0f} "
          f"({100*r['total_failure_rate_wilson_ci_lower']:.0f}-{100*r['total_failure_rate_wilson_ci_upper']:.0f})",
          f"{r['gate_blend_events_per_1000_steps_mean']:.3f} +/- {r['gate_blend_events_per_1000_steps_std']:.3f}"]
         for r in agg_rows])
    print(f"[OK] Table 2 -> {out('table2_summary_by_dopamine.md')}")

    if stat_rows:
        # NOTE: the "Significant" verdict column was dropped after review
        # -- it silently collapsed two separate tests (Fisher, Mann-
        # Whitney) into one flat yes/no, which hid real disagreements
        # between the tests (e.g. D=0.05 and D=0.5 each significant in
        # one test but not the other). Leaving just the two p-values lets
        # the reader judge against p<0.05 directly, without a pre-baked
        # verdict.
        write_markdown_table(
            out("table3_significance.md"),
            f"Table 3. Significance vs. baseline dopamine (D={baseline_dopamine}), "
            f"Holm-Bonferroni corrected across {len(stat_rows)} comparisons. "
            f"Significant at p<0.05 when the corrected value is below that threshold.",
            ["Dopamine", "Fisher OR", "Fisher p (corrected)", "Mann-Whitney p (corrected)"],
            [[r["dopamine"], f"{r['fisher_odds_ratio']:.3g}", f"{r['fisher_p_holm_corrected']:.3g}",
              f"{r['mannwhitney_p_holm_corrected']:.3g}"]
             for r in stat_rows])
        print(f"[OK] Table 3 -> {out('table3_significance.md')}")


if __name__ == "__main__":
    main()
