"""3개 타겟(SATFIN/HELPFUL/PARTYLR, exp009~011) 결과를 pooled(n=300)로 합쳐 통계 검정."""
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

paths = {
    "SATFIN": "../exp009_phase1_satfin_n100/output_predictions.csv",
    "HELPFUL": "../exp010_phase1_helpful_n100/output_predictions.csv",
    "PARTYLR": "../exp011_phase1_partylr_n100/output_predictions.csv",
}
dfs = []
for t, p in paths.items():
    d = pd.read_csv(p)
    d["target"] = t
    d["uid"] = d["target"] + "_" + d["respid"].astype(str)
    dfs.append(d)
res = pd.concat(dfs, ignore_index=True)
arms = ["arm0_baseline", "arm1_corr", "arm2_embed", "arm3_theory", "arm4_random"]

pivot_correct = res.pivot(index="uid", columns="arm", values="correct")
pivot_dist = res.pivot(index="uid", columns="arm", values="distance")

print("=== POOLED (3 targets, n=300) accuracy + 95% CI + MAE ===")
for a in arms:
    p = pivot_correct[a].mean()
    n = pivot_correct[a].notna().sum()
    lo, hi = proportion_confint(pivot_correct[a].sum(), n, method="wilson")
    mae = pivot_dist[a].mean()
    print(f"{a}: acc={p:.3f} CI=({lo:.3f},{hi:.3f})  MAE={mae:.3f}")

print("\n=== McNemar vs arm0 (pooled) ===")
for a in arms:
    if a == "arm0_baseline":
        continue
    t = pd.crosstab(pivot_correct["arm0_baseline"], pivot_correct[a])
    print(f"arm0 vs {a}: p={mcnemar(t, exact=True).pvalue:.4f}")

print("\n=== McNemar: arm3(항상 최하위) vs others (pooled) ===")
for a in ["arm0_baseline", "arm1_corr", "arm2_embed", "arm4_random"]:
    t = pd.crosstab(pivot_correct["arm3_theory"], pivot_correct[a])
    print(f"arm3 vs {a}: p={mcnemar(t, exact=True).pvalue:.4f}")

print("\n=== Wilcoxon MAE vs arm0 (pooled) ===")
for a in arms:
    if a == "arm0_baseline":
        continue
    stat, p = wilcoxon(pivot_dist["arm0_baseline"], pivot_dist[a])
    print(f"arm0 vs {a}: p={p:.4f} diff={(pivot_dist['arm0_baseline']-pivot_dist[a]).mean():+.3f}")

print("\n=== Wilcoxon MAE: arm3 vs others (pooled) ===")
for a in ["arm0_baseline", "arm1_corr", "arm2_embed", "arm4_random"]:
    stat, p = wilcoxon(pivot_dist["arm3_theory"], pivot_dist[a])
    print(f"arm3 vs {a}: p={p:.4f} diff={(pivot_dist['arm3_theory']-pivot_dist[a]).mean():+.3f}")
