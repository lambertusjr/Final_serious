import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def results_to_long_df(testing_results: dict) -> pd.DataFrame:
    """
    testing_results[model][metric] -> list of per-run values
    Returns a long DataFrame: columns = ['model','metric','run','value'].
    """
    records = []
    for model, metrics in testing_results.items():
        for metric_name, values in metrics.items():
            for i, v in enumerate(values):
                # Guard against Nones or NaNs in case some runs failed
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                records.append({
                    'model': model,
                    'metric': metric_name,
                    'run': i + 1,
                    'value': float(v)
                })
    return pd.DataFrame.from_records(records)


def summarise_long_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary (mean, sd, n, 95% CI half-width) by model × metric.
    """
    if df_long.empty:
        return pd.DataFrame(columns=['model','metric','n','mean','sd','ci95'])
    grouped = df_long.groupby(['model','metric'])['value']
    summary = grouped.agg(n='count', mean='mean', sd='std').reset_index()
    # Normal approximation is fine for n≈30 (your case)
    summary['ci95'] = 1.96 * (summary['sd'] / np.sqrt(summary['n']).replace(0, np.nan))
    return summary


def formatted_wide_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a wide table: one row per model, columns are metrics,
    with each cell formatted as 'mean ± sd [CI95]' to two/three decimals.
    """
    if summary_df.empty:
        return pd.DataFrame()
    tmp = summary_df.copy()
    tmp['cell'] = (
        tmp['mean'].map(lambda x: f"{x:.3f}")
        + " ± "
        + tmp['sd'].fillna(0).map(lambda x: f"{x:.3f}")
        + " ["
        + tmp['ci95'].fillna(0).map(lambda x: f"{x:.3f}")
        + "]"
    )
    wide = tmp.pivot(index='model', columns='metric', values='cell').sort_index()
    # Optional: sort columns in a sensible order if present
    desired_order = [
        'precision_weighted', 'precision_illicit',
        'recall_weighted', 'recall_illicit',
        'f1_weighted', 'f1_illicit'
    ]
    cols = [c for c in desired_order if c in wide.columns] + [c for c in wide.columns if c not in desired_order]
    return wide[cols]

def produce_tables(testing_results):
    # Build DataFrames
    df_long = results_to_long_df(testing_results)
    df_summary = summarise_long_df(df_long)
    df_wide = formatted_wide_table(df_summary)

    # Console views
    print("\n=== Per-run (long) results ===")
    print(df_long.head().to_string(index=False))

    print("\n=== Summary by model × metric ===")
    print(df_summary.to_string(index=False))

    print("\n=== Paper-ready wide table (mean ± sd [CI95]) ===")
    print(df_wide.to_string())

    # Optional: export for paper
    df_wide.to_csv("testing_summary.csv")
    # If you need LaTeX for your thesis/paper:
    with open("testing_summary.tex", "w") as f:
        f.write(df_wide.to_latex(escape=False))
    return df_long, df_summary, df_wide
        
def boxplots_by_metric(df_long: pd.DataFrame):
    """
    For each metric, draws a separate box plot across models.
    """
    if df_long.empty:
        print("No data to plot.")
        return

    for metric in df_long['metric'].unique():
        subset = df_long[df_long['metric'] == metric]
        if subset.empty:
            continue

        plt.figure(figsize=(8, 5))
        # Ensure consistent model order
        models = sorted(subset['model'].unique())
        data = [subset[subset['model'] == m]['value'].values for m in models]

        plt.boxplot(data, labels=models, showmeans=True)
        plt.title(f"Distribution of {metric} across models")
        plt.ylabel(metric.replace('_', ' '))
        plt.xlabel("Model")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
def bar_means_with_ci(df_summary: pd.DataFrame, metric: str):
    """
    Bar chart of mean ± 95% CI for the selected metric across models.
    """
    sub = df_summary[df_summary['metric'] == metric].copy()
    if sub.empty:
        print(f"No summary for metric '{metric}'.")
        return

    sub = sub.sort_values('model')
    x = np.arange(len(sub))
    y = sub['mean'].values
    yerr = sub['ci95'].fillna(0).values

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, sub['model'].tolist())
    plt.ylim(0, min(1.0, max(1.0, (y + yerr).max() * 1.1)))  # metrics are typically in [0,1]
    plt.title(f"Mean ± 95% CI for {metric}")
    plt.ylabel(metric.replace('_', ' '))
    plt.xlabel("Model")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()