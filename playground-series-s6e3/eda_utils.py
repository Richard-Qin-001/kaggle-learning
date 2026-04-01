    # EDA Utils
    # Copyright (C) 2026  Richard Qin

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def get_feature_groups(df, target=None, categorical_nunique_threshold=15):
    """
    Infer numeric and categorical features for general-purpose EDA.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Low-cardinality numeric columns are often class-like variables in tabular tasks.
    for col in numeric_cols.copy():
        if df[col].nunique(dropna=True) <= categorical_nunique_threshold:
            categorical_cols.append(col)
            numeric_cols.remove(col)

    if target is not None:
        numeric_cols = [c for c in numeric_cols if c != target]
        categorical_cols = [c for c in categorical_cols if c != target]

    return {
        "numeric": sorted(set(numeric_cols)),
        "categorical": sorted(set(categorical_cols)),
    }


def basic_summary(df, df_name="Dataset"):
    """
    Print high-level dataset stats and return a column-wise profile DataFrame.
    """
    print(f"--- {df_name} Basic Summary ---")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Duplicates: {df.duplicated().sum()}")
    print("-" * 35)

    summary = pd.DataFrame(
        {
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "% Missing": (df.isnull().sum() / len(df)) * 100,
            "Unique Values": df.nunique(dropna=True),
        }
    ).sort_values(by="% Missing", ascending=False)

    display(summary)
    return summary


def missing_values_report(df, top_n=None):
    """
    Return missing value counts/ratios sorted by missing ratio.
    """
    report = pd.DataFrame(
        {
            "Missing Values": df.isnull().sum(),
            "% Missing": (df.isnull().sum() / len(df)) * 100,
        }
    ).sort_values(by="% Missing", ascending=False)

    report = report[report["Missing Values"] > 0]
    if top_n is not None:
        report = report.head(top_n)

    if report.empty:
        print("No missing values found.")
    else:
        display(report)

    return report


def numeric_profile(df, features=None, quantiles=(0.01, 0.05, 0.5, 0.95, 0.99)):
    """
    Extended numeric profile with skewness and kurtosis.
    """
    if features is None:
        features = df.select_dtypes(include=["number"]).columns.tolist()

    if not features:
        print("No numerical features found.")
        return pd.DataFrame()

    desc = df[features].describe(percentiles=quantiles).T
    desc["skew"] = df[features].skew(numeric_only=True)
    desc["kurtosis"] = df[features].kurtosis(numeric_only=True)
    display(desc)
    return desc


def categorical_profile(df, features=None, top_n=10):
    """
    Show category cardinality and top value frequencies for each categorical feature.
    """
    if features is None:
        features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not features:
        print("No categorical features found.")
        return {}

    profile = {}
    for feature in features:
        vc = df[feature].value_counts(dropna=False).head(top_n)
        freq_df = pd.DataFrame(
            {
                "count": vc,
                "%": (vc / len(df) * 100).round(2),
            }
        )
        profile[feature] = {
            "nunique": df[feature].nunique(dropna=True),
            "top_values": freq_df,
        }
        print(f"\n--- {feature} (nunique={profile[feature]['nunique']}) ---")
        display(freq_df)

    return profile


def plot_target_distribution(df, target):
    """
    Plot target distribution for classification and regression tasks.
    """
    if target not in df.columns:
        print(f"Target column '{target}' not found.")
        return

    plt.figure(figsize=(8, 4))
    target_is_classification = df[target].nunique(dropna=True) <= 15
    if target_is_classification:
        order = df[target].value_counts(dropna=False).index
        sns.countplot(data=df, x=target, order=order)
        plt.title(f"Target Distribution ({target})")
        plt.ylabel("Count")
    else:
        sns.histplot(df[target], kde=True, bins=30)
        plt.title(f"Target Distribution ({target})")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_numerical_distributions(
    df,
    features=None,
    bins=30,
    kde=True,
    cols_per_row=3,
    max_plots=12,
):
    """
    Plot histogram distributions for numeric variables.
    """
    if features is None:
        features = df.select_dtypes(include=["number"]).columns.tolist()

    features = features[:max_plots]
    num_features = len(features)
    if num_features == 0:
        print("No numerical features found.")
        return

    rows = int(np.ceil(num_features / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3.8 * rows))
    axes = np.array(axes).flatten()

    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, bins=bins, kde=kde, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribution of {feature}", fontsize=11)
        axes[i].set_xlabel("")

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df,
    features=None,
    top_n=10,
    cols_per_row=3,
    max_plots=9,
    normalize=False,
):
    """
    Plot bar charts for categorical features with top-N categories.
    """
    if features is None:
        features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    features = features[:max_plots]
    num_features = len(features)
    if num_features == 0:
        print("No categorical features found.")
        return

    rows = int(np.ceil(num_features / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
    axes = np.array(axes).flatten()

    for i, feature in enumerate(features):
        vc = df[feature].value_counts(dropna=False).head(top_n)
        plot_df = vc.rename_axis(feature).reset_index(name="count")
        if normalize:
            plot_df["count"] = plot_df["count"] / plot_df["count"].sum()

        sns.barplot(data=plot_df, y=feature, x="count", ax=axes[i], palette="Set2")
        axes[i].set_title(f"Top {top_n} categories: {feature}", fontsize=11)
        axes[i].set_xlabel("Ratio" if normalize else "Count")

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
    df,
    features=None,
    method="spearman",
    figsize=None,
    annot=False,
):
    """
    Draw a correlation heatmap for numeric features.
    """
    if features is None:
        df_num = df.select_dtypes(include=["number"])
    else:
        df_num = df[features].select_dtypes(include=["number"])

    if df_num.empty:
        print("No numerical features to correlate.")
        return pd.DataFrame()

    corr = df_num.corr(method=method)
    if figsize is None:
        size = min(max(8, len(corr.columns) * 0.45), 22)
        figsize = (size, size)

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.7},
    )
    plt.title(f"Correlation Matrix ({method.capitalize()})", fontsize=14)
    plt.tight_layout()
    plt.show()
    return corr


def plot_target_vs_numerical(df, target, features=None, cols_per_row=3, max_plots=9):
    """
    Plot numeric features against target.
    Classification target: boxplot; Regression target: scatterplot.
    """
    if target not in df.columns:
        print(f"Target column '{target}' not found.")
        return

    if features is None:
        features = [c for c in df.select_dtypes(include=["number"]).columns if c != target]

    features = features[:max_plots]
    num_features = len(features)
    if num_features == 0:
        print("No numerical features available for target relationship plot.")
        return

    is_classification = df[target].nunique(dropna=True) <= 15
    rows = int(np.ceil(num_features / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
    axes = np.array(axes).flatten()

    for i, feature in enumerate(features):
        if is_classification:
            sns.boxplot(data=df, x=target, y=feature, ax=axes[i], palette="Set3")
        else:
            sns.scatterplot(data=df, x=feature, y=target, ax=axes[i], s=20, alpha=0.7)
        axes[i].set_title(f"{feature} vs {target}", fontsize=11)

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_target_vs_categorical(df, target, features=None, top_n=8, cols_per_row=3, max_plots=6):
    """
    Plot target relationship with categorical features.
    Classification target: stacked proportion chart.
    Regression target: boxplot of target by category.
    """
    if target not in df.columns:
        print(f"Target column '{target}' not found.")
        return

    if features is None:
        features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    features = [f for f in features if f != target][:max_plots]
    num_features = len(features)
    if num_features == 0:
        print("No categorical features available for target relationship plot.")
        return

    is_classification = df[target].nunique(dropna=True) <= 15
    rows = int(np.ceil(num_features / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
    axes = np.array(axes).flatten()

    for i, feature in enumerate(features):
        top_categories = df[feature].value_counts(dropna=False).head(top_n).index
        temp = df[df[feature].isin(top_categories)].copy()
        temp[feature] = temp[feature].astype(str)

        if is_classification:
            ct = pd.crosstab(temp[feature], temp[target], normalize="index")
            ct.plot(kind="bar", stacked=True, ax=axes[i], legend=False, colormap="Set2")
            axes[i].set_ylabel("Proportion")
        else:
            sns.boxplot(data=temp, x=feature, y=target, ax=axes[i], palette="Set2")
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].set_ylabel(target)
        axes[i].set_title(f"{target} by {feature}", fontsize=11)

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df, features=None, threshold=1.5, return_index=False):
    """
    Detect outliers by IQR rule and return summary report.
    """
    if features is None:
        features = df.select_dtypes(include=["number"]).columns.tolist()

    if not features:
        print("No numerical features found for outlier detection.")
        return pd.DataFrame() if not return_index else ({}, pd.DataFrame())

    outlier_indices = {}
    summary_data = []

    for feature in features:
        series = df[feature].dropna()
        if series.empty:
            outlier_indices[feature] = []
            summary_data.append({"Feature": feature, "Outliers Count": 0, "% Outliers": 0.0})
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_indices[feature] = []
            summary_data.append({"Feature": feature, "Outliers Count": 0, "% Outliers": 0.0})
            continue

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index.tolist()
        outlier_indices[feature] = outliers
        summary_data.append(
            {
                "Feature": feature,
                "Outliers Count": len(outliers),
                "% Outliers": (len(outliers) / len(df)) * 100,
            }
        )

    summary_df = pd.DataFrame(summary_data).sort_values(by="Outliers Count", ascending=False)
    display(summary_df[summary_df["Outliers Count"] > 0])

    if return_index:
        return outlier_indices, summary_df
    return summary_df


def run_quick_eda(df, target=None, df_name="Dataset"):
    """
    One-call EDA entry point for fast checks across projects.
    """
    groups = get_feature_groups(df, target=target)
    basic_summary(df, df_name=df_name)
    missing_values_report(df)
    numeric_profile(df, features=groups["numeric"][:30])

    if groups["categorical"]:
        categorical_profile(df, features=groups["categorical"][:8], top_n=8)

    plot_numerical_distributions(df, features=groups["numeric"], max_plots=12)

    if groups["categorical"]:
        plot_categorical_distributions(df, features=groups["categorical"], max_plots=6)

    if target is not None and target in df.columns:
        plot_target_distribution(df, target)
        plot_target_vs_numerical(df, target, features=groups["numeric"], max_plots=9)
        if groups["categorical"]:
            plot_target_vs_categorical(df, target, features=groups["categorical"], max_plots=6)
