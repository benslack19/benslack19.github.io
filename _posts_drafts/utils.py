"""Common functions in blog posts."""

import arviz as az
import graphviz as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def draw_causal_graph(
    edge_list, node_props=None, edge_props=None, graph_direction="UD"
):
    """Utility to draw a causal (directed) graph
    Taken from: https://github.com/dustinstansbury/statistical-rethinking-2023/blob/a0f4f2d15a06b33355cf3065597dcb43ef829991/utils.py#L52-L66

    """
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props[e] if e in edge_props else {}
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


def standardize(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


def plot_comparison(
    trace0,
    trace1,
    var_name,
    xlabel,
    ylabel,
    meta_df=None,
    cluster_col=None,
    hue=None,
    title_prefix=None,
):
    """Make 2D scatter plots of mean and SD

    Note: Use with pymc output.

    Parameters
    ----------
    trace0
        Trace result object on x-axis
    trace1
        Trace result object on y-axis
    var
        Variable of trace objects to compare
    xlabel
        X-axis label
    ylabel
        Y-axis label
    meta_df
        A dataframe containing values for a pheno_col
    cluster_col
        Cluster column name to represent individual clusters (e.g. patients)
    hue
        Phenotype column name of a categorical variable, for hue
    title_prefix
        Title prefix

    Returns
    -------
    None

    """

    # Create dfs for plotting
    df_trace0 = az.summary(trace0, var_names=[var_name])[["mean", "sd"]]
    df_trace1 = az.summary(trace1, var_names=[var_name])[["mean", "sd"]]

    def _format_df_for_seaborn(az_summary_col, cluster_col):
        df = pd.concat(
            [
                df_trace0[az_summary_col]
                .reset_index()
                .rename(columns={az_summary_col: xlabel}),
                df_trace1[az_summary_col]
                .reset_index()
                .rename(columns={az_summary_col: ylabel})
                .drop(columns="index"),
            ],
            axis=1,
        )

        if cluster_col and hue:
            df[cluster_col] = (
                df["index"].str.lstrip(f"{var_name}[").str.rstrip("]").astype(int)
            )

            df = df.merge(meta_df[[cluster_col, hue]].drop_duplicates(), on=cluster_col)

        data_min = df[[xlabel, ylabel]].min().min()
        data_max = df[[xlabel, ylabel]].max().max()

        return df, data_min, data_max

    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    df_mean, mean_min, mean_max = _format_df_for_seaborn(
        "mean", cluster_col=cluster_col
    )
    df_sd, sd_min, sd_max = _format_df_for_seaborn("sd", cluster_col=cluster_col)

    sns.scatterplot(data=df_mean, x=xlabel, y=ylabel, hue=hue, marker="$\circ$", ax=ax0)
    sns.scatterplot(data=df_sd, x=xlabel, y=ylabel, hue=hue, marker="$\circ$", ax=ax1)

    # Add dashed line and set the limits to be the same for both axes
    ax0.plot([0, 1], [0, 1], transform=ax0.transAxes, linestyle="dashed", color="gray")
    ax0.set_xlim(mean_min * 0.95, mean_max * 1.05)
    ax0.set_ylim(mean_min * 0.95, mean_max * 1.05)
    ax0.set_title(f"{title_prefix}\n(mean)")

    ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, linestyle="dashed", color="gray")
    ax1.set_xlim(sd_min * 0.95, sd_max * 1.05)
    ax1.set_ylim(sd_min * 0.95, sd_max * 1.05)
    ax1.set_title(f"{title_prefix}\n(SD)")

    f.tight_layout()
