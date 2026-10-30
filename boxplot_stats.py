import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

metric_to_plot = "Test"
BEST_CRITERION = "mean"
ALPHA = 0.05

def holm_correction(pvals: np.ndarray) -> np.ndarray:

    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)

    for i, idx in enumerate(order):
        adj[idx] = (m - i) * pvals[idx]

    sorted_adj = adj[order]
    sorted_adj = np.maximum.accumulate(sorted_adj)
    sorted_adj = np.clip(sorted_adj, 0.0, 1.0)
    adj[order] = sorted_adj
    return adj

def bonferroni_correction(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    return np.clip(pvals * len(pvals), 0.0, 1.0)

root_dir = Path(__file__).resolve().parent

files = list(root_dir.glob('**/*results_with_seeds.csv'))
data_grouped = {'Acc_test':[],'Configuration':[],'Classifier':[]}
data_ungrouped = {'Acc_test':[],'Configuration':[],'Classifier':[]}
for path in files:
    path_string = str(path)
    position=(path_string.find('results_with_seeds'))
    name_classifier =''
    data = pd.read_csv(path_string)

    data_grouped['Acc_test'].append(data['Acc_Test'].values)
    data_grouped['Configuration'].append(data['Configuration'].values)
    data_ungrouped['Acc_test'].extend(data['Acc_Test'].values)
    data_ungrouped['Configuration'].extend(data['Configuration'].values)
    for i in range(0,position-1):
        name_classifier+=path_string[i]
        if path_string[i] == "\\":
            name_classifier = ""
    data_grouped['Classifier'].append(name_classifier)
    data_ungrouped['Classifier'].extend([name_classifier]*len(data))

data_grouped = pd.DataFrame(data_grouped)
data_ungrouped = pd.DataFrame(data_ungrouped)

print(data_grouped.head(15))
data_grouped = data_grouped.dropna()
personalized_order =['polynomialclass', 'naivebayes',  'logisticlass',  'perceptron', 'svmclass','randomforest','FNN','CNN1', 'CNN2', 'CNN3', 'Transformer1', 'Transformer2', 'Transformer3','CNN1Transformer1','CNN2Transformer1', 'CNN2Transformer2',  'CNN3Transformer1', 'CNN3Transformer2', 'CNN3Transformer3']

data_grouped['Classifier'] = pd.Categorical(
    data_grouped['Classifier'],
    categories=personalized_order,
    ordered=True
)

data_grouped = data_grouped.sort_values(by='Classifier')
classifiers = data_grouped['Classifier'].drop_duplicates().values
data_by_classifier = data_grouped['Acc_test'].values

plt.figure(figsize=(max(10, 0.8 * len(classifiers)), 5))
plt.boxplot(data_by_classifier, labels=classifiers, showmeans=True, showfliers=True)
plt.ylabel("Accuracy")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()

samples = {}
for i in range(len(classifiers)):
    samples[classifiers[i]] = data_by_classifier[i]
print(samples)

summary = (
    data_ungrouped.groupby("Classifier")["Acc_test"]
    .agg(n="count", mean="mean", median="median", max="max", min="min", std="std")
    .sort_values(["mean", "median", "max"], ascending=False)
)

ci_margin = stats.t.ppf(1 - ALPHA / 2, summary["n"] - 1) * summary["std"] / np.sqrt(summary["n"])

summary["CI 95% Lower"] = summary["mean"] - ci_margin
summary["CI 95% Upper"] = summary["mean"] + ci_margin

metric_to_plot = "Accuracy"
print("\n=== Summary by classifier ===")
print(f"Metric: {metric_to_plot}")
print(summary.to_string())

groups = [samples[c] for c in samples.keys()]
H, p_kw = stats.kruskal(*groups, nan_policy="omit")

print("\n=== Kruskal-Wallis H-test (across classifiers) ===")
print(f"Metric: {metric_to_plot}")
print(f"H statistic = {H:.6g}")
print(f"p-value     = {p_kw:.6g}")

criterion_func = {
    "mean":   lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "max":    lambda x: float(np.max(x)),
}

if BEST_CRITERION not in criterion_func:
    raise ValueError("BEST_CRITERION must be one of: 'mean', 'median', 'max'.")

best = max(samples.keys(), key=lambda c: criterion_func[BEST_CRITERION](samples[c]))

print("\n=== Best classifier (reference) ===")
print(f"Selection criterion = {BEST_CRITERION}")
print(f"Best classifier      = {best}")
print(f"Mean accuracy        = {np.mean(samples[best]):.6g}")
print(f"Median accuracy      = {np.median(samples[best]):.6g}")
print(f"Max accuracy         = {np.max(samples[best]):.6g}")
print(f"n                   = {len(samples[best])}")

rows = []
raw_pvals = []
compared = []

for c in samples.keys():
    if c == best:
        continue

    x = samples[best]
    y = samples[c]

    U, p_u = stats.mannwhitneyu(x, y, alternative="greater")
    raw_pvals.append(p_u)
    compared.append(c)

    rows.append({
        "Metric": metric_to_plot,
        "Reference (Best)": best,
        "Compared classifier": c,
        "U statistic": U,
        "p-value (one-sided, best>other)": p_u,
        "Mean(best)": np.mean(x),
        "Mean(other)": np.mean(y),
        "Median(best)": np.median(x),
        "Median(other)": np.median(y),
        "Max(best)": np.max(x),
        "Max(other)": np.max(y),
        "n(best)": len(x),
        "n(other)": len(y),
    })

raw_pvals = np.asarray(raw_pvals, dtype=float)

holm_p = holm_correction(raw_pvals)
bonf_p = bonferroni_correction(raw_pvals)

for i, r in enumerate(rows):
    r["p-value Holm"] = holm_p[i]
    r["p-value Bonferroni"] = bonf_p[i]
    r["Significant (Holm, alpha=0.05)"] = holm_p[i] < ALPHA

posthoc_df = pd.DataFrame(rows).sort_values("p-value (one-sided, best>other)")

print("\n=== Post-hoc pairwise (Mann-Whitney U, one-sided: best>other) ===")
print(posthoc_df.to_string(index=False))

out_csv = f"posthoc_mannwhitney_{metric_to_plot.replace(' ', '_').lower()}_best_{BEST_CRITERION}.csv"
posthoc_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

REFERENCE_CLASSIFIER = "CNN1"

paired_classifiers = list(samples.keys())
paired_groups = [np.asarray(samples[c], dtype=float) for c in paired_classifiers]
if any(len(x) != 11 or not np.isfinite(x).all() for x in paired_groups):
    raise ValueError("Paired tests require 11 finite accuracies per classifier, aligned by seed (0-10).")
if REFERENCE_CLASSIFIER not in samples:
    raise ValueError(f"Reference classifier '{REFERENCE_CLASSIFIER}' not found.")

print("\n=== Friedman test (paired by random seed across classifiers) ===")
print("H0: Classifier accuracy ranks are equal across matched seeds.")
print("H1: At least one classifier differs in its accuracy ranks.")
chi2_friedman, p_friedman = stats.friedmanchisquare(*paired_groups)
print(f"Chi-square statistic = {chi2_friedman:.6g}")
print(f"p-value              = {p_friedman:.6g}")

print("\n=== Wilcoxon signed-rank (two-sided, paired by seed) ===")
print(f"Fixed reference: {REFERENCE_CLASSIFIER}")
print("H0: The median paired accuracy difference is zero (assuming symmetric differences).")
print("H1: The median paired accuracy difference is nonzero.")
print("These results are exploratory if the reference was selected after inspecting results.")

wilcoxon_rows = []
wilcoxon_raw_p = []
ref_acc = np.asarray(samples[REFERENCE_CLASSIFIER], dtype=float)
for classifier in paired_classifiers:
    if classifier == REFERENCE_CLASSIFIER:
        continue
    other_acc = np.asarray(samples[classifier], dtype=float)
    diff = ref_acc - other_acc
    if np.all(diff == 0):
        W, p_value = 0.0, 1.0
    else:
        W, p_value = stats.wilcoxon(
            ref_acc, other_acc, alternative="two-sided", zero_method="wilcox"
        )
    wilcoxon_raw_p.append(p_value)
    wilcoxon_rows.append({
        "Reference": REFERENCE_CLASSIFIER,
        "Compared classifier": classifier,
        "W statistic": W,
        "p-value (two-sided)": p_value,
        "Mean(reference)": np.mean(ref_acc),
        "Mean(other)": np.mean(other_acc),
        "Mean paired difference": np.mean(diff),
        "Median paired difference": np.median(diff),
        "n paired seeds": len(diff),
    })

wilcoxon_holm_p = holm_correction(np.asarray(wilcoxon_raw_p, dtype=float))
for i, row in enumerate(wilcoxon_rows):
    row["p-value Holm"] = wilcoxon_holm_p[i]
    row["Significant (Holm, alpha=0.05)"] = wilcoxon_holm_p[i] < ALPHA

wilcoxon_df = pd.DataFrame(wilcoxon_rows).sort_values("p-value (two-sided)")
print(wilcoxon_df.to_string(index=False))
wilcoxon_csv = f"posthoc_wilcoxon_{metric_to_plot.replace(' ', '_').lower()}_reference_{REFERENCE_CLASSIFIER}.csv"
wilcoxon_df.to_csv(wilcoxon_csv, index=False)
print(f"\nSaved: {wilcoxon_csv}")
