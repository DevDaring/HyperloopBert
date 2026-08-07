"""
Why does the official CrowS-Pairs rule reverse the sign of the architecture
contrast, when the full-sentence and shared-token rules do not?

Three things differ between this project's two PLL rules and the benchmark's
released implementation, and the paper previously named only two of them:

  1. TOKEN SET      all positions            vs  the shared (unmodified) tokens
  2. AGGREGATION    mean over scored tokens  vs  sum over scored tokens
  3. DIRECTION      sent_more is always      vs  for the 218 `antistereo` pairs
                    treated as the               the roles are swapped, so
                    stereotypical member         sent_LESS is the stereotypical
                                                 member

Difference 3 is the one the paper did not state. It is not cosmetic: it changes
the sign of the per-item effect on 218 of 1508 pairs (14.5%).

This script crosses aggregation with the direction convention on a single fixed
token set (the shared tokens recovered by difflib, i.e. the official one), so
the reversal can be attributed to one factor rather than to their bundle.

CPU only, no training, reads the already-scored per-item files.
"""
import os
import numpy as np
import pandas as pd

ROOT = "/home/Debz/Research/HyperloopBert"
OFF = os.path.join(ROOT, "analysis/fire2026/out/official")
OUT = os.path.join(ROOT, "analysis/fire2026/out")
SEED = 20260729
B = 10_000
ARCHS = ["Vanilla", "Looped", "ALBERT", "Hyperloop"]


def per_item(df, aggregation, direction_aware):
    """Per-item effect e = PLL(stereotypical) - PLL(anti-stereotypical)."""
    more, less = ((df.norm_more, df.norm_less) if aggregation == "mean"
                  else (df.score_more, df.score_less))
    e = (more - less).to_numpy(float)
    if direction_aware:                      # swap roles on the antistereo pairs
        e = np.where((df.direction == "antistereo").to_numpy(), -e, e)
    return e


def perm_p(diff, rng, b=B):
    """Paired sign-flip permutation test, Phipson-Smyth (b+1)/(m+1)."""
    obs = abs(diff.mean())
    flips = rng.choice([-1.0, 1.0], size=(b, diff.size))
    null = np.abs((flips * diff).mean(axis=1))
    return (int((null >= obs).sum()) + 1) / (b + 1)


def main():
    data = {a: pd.read_csv(os.path.join(OFF, f"{a}_2p20.csv")) for a in ARCHS}
    n = len(data["Vanilla"])
    n_anti = int((data["Vanilla"].direction == "antistereo").sum())
    print(f"CrowS-Pairs pairs scored at the deepest matched point : {n}")
    print(f"  of which labelled `antistereo` by the dataset       : {n_anti} "
          f"({100 * n_anti / n:.1f}%)")
    print(f"  mean shared tokens scored per sentence              : "
          f"{data['Vanilla'].n_shared_more.mean():.1f}\n")

    rng = np.random.default_rng(SEED)
    rows = []
    for aggregation in ("mean", "sum"):
        for direction_aware in (False, True):
            base = per_item(data["Vanilla"], aggregation, direction_aware)
            for arch in ARCHS[1:]:
                d = base - per_item(data[arch], aggregation, direction_aware)
                boot = np.array([d[rng.integers(0, d.size, d.size)].mean()
                                 for _ in range(B)])
                rows.append(dict(
                    aggregation=aggregation,
                    direction_convention=("official (roles swapped on antistereo)"
                                          if direction_aware
                                          else "sent_more always stereotypical"),
                    contrast=f"Vanilla vs {arch}",
                    delta=d.mean(),
                    ci_low=np.percentile(boot, 2.5),
                    ci_high=np.percentile(boot, 97.5),
                    p_perm=perm_p(d, rng)))

    res = pd.DataFrame(rows)
    dest = os.path.join(OUT, "scorer_decomposition.csv")
    res.to_csv(dest, index=False)

    print("Architecture contrast on the official token set, crossing the two "
          "factors\n(positive = the unshared baseline records the higher effect)\n")
    for (agg, conv), g in res.groupby(["aggregation", "direction_convention"],
                                      sort=False):
        print(f"  aggregation={agg:4s}  direction={conv}")
        for _, r in g.iterrows():
            star = "*" if r.p_perm < 0.05 else " "
            print(f"      {r.contrast:22s} {r.delta:+.5f} "
                  f"[{r.ci_low:+.5f}, {r.ci_high:+.5f}]  p={r.p_perm:.4f}{star}")
        print()

    print("sign of the contrast under each direction convention "
          "(both aggregations agree):")
    for contrast, g in res.groupby("contrast", sort=False):
        signs = {conv: sorted({int(np.sign(v)) for v in gg.delta})
                 for conv, gg in g.groupby("direction_convention", sort=False)}
        shown = "   ".join(f"{c.split(' (')[0]}: "
                           f"{'/'.join('+' if s > 0 else '-' for s in v)}"
                           for c, v in signs.items())
        print(f"  {contrast:22s} {shown}")
    print()
    print(f"written to {dest}")


if __name__ == "__main__":
    main()
