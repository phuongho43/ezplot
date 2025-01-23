from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from ezplot.process import calc_tspan_aves
from ezplot.style import PALETTE, STYLE


def draw_sigdiff(ax, df, cg1, cg2):
    for line in ax.lines:
        X = line.get_xdata()
        n_groups = df["group"].nunique()
        n_classes = df["class"].nunique()
        if len(X) == n_classes * n_groups:
            X = np.reshape(X, (n_classes, n_groups))
            break
    x1 = X[*cg1]
    x2 = X[*cg2]
    xm = (x1 + x2) / 2
    cg1_vals = df.loc[(df["class"] == cg1[0]) & (df["group"] == cg1[1]), "response"]
    cg2_vals = df.loc[(df["class"] == cg2[0]) & (df["group"] == cg2[1]), "response"]
    dy = (df["response"].max() - df["response"].min()) / 25
    y = max(cg1_vals.max(), cg2_vals.max()) + dy
    pval = sp.stats.ttest_ind(cg1_vals, cg2_vals).pvalue
    if pval <= 0.001:
        text = "ns"
    elif pval <= 0.01:
        text = "**"
    elif pval <= 0.05:
        text = "*"
    else:
        text = "ns"
    bar_x = [x1, x1, x2, x2]
    bar_y = [y, y + dy, y + dy, y]
    ax.plot(bar_x, bar_y, color="#212121", lw=8, zorder=10.1)
    ax.text(xm, y + dy, text, ha="center", va="bottom", zorder=10.2)


def plot_rcgy(fig_fp, rcgy_df, xlabel, ylabel, group_labels, class_labels=None, sigdiff_cgs=None, palette=PALETTE, rc_params=STYLE, figsize=(24, 16)):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        if class_labels:
            hue = "class"
            dodge1 = 0.4
            dodge2 = 0.4
        else:
            hue = "group"
            dodge1 = False
            dodge2 = False
        sns.stripplot(
            ax=ax,
            data=rcgy_df,
            x="group",
            y="response",
            hue=hue,
            dodge=dodge1,
            jitter=0.1,
            palette=palette,
            size=2 * rc_params["lines.markersize"],
            linewidth=rc_params["lines.markeredgewidth"],
            edgecolor=rc_params["lines.markeredgecolor"],
            alpha=0.8,
            zorder=2.1,
        )
        sns.pointplot(
            ax=ax,
            data=rcgy_df,
            x="group",
            y="response",
            hue=hue,
            dodge=dodge2,
            errorbar=("se", 1.96),
            palette="dark:#212121",
            markers=".",
            markersize=rc_params["lines.markersize"],
            err_kws={"linewidth": 10},
            capsize=0.2,
            zorder=2.2,
        )
        for cg1, cg2 in sigdiff_cgs:
            draw_sigdiff(ax=ax, df=rcgy_df, cg1=cg1, cg2=cg2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels)
        if class_labels:
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, class_labels, loc="best")
        else:
            ax.get_legend().remove()
        fig.savefig(fig_fp)
    plt.close("all")


def main():
    ## Figure 2J ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2j.png"
    y_csv_fp = "/home/phuong/data/phd-project/1--biosensor/5--sparser/results/y.csv"
    rty_df = pd.read_csv(y_csv_fp)
    group_labels = [
        "None\nInput",
        "Sparse\nInput",
        "Dense\nInput",
    ]
    palette = ["#34495E", "#EA822C", "#8069EC"]
    ylabel = r"$\mathbf{Ave\ \Delta F/F_{0}}$"
    tspans = [[0, 60], [60, 120], [120, 155]]
    aves_df = calc_tspan_aves(rty_df, tspans=tspans)
    aves_df["class"] = np.zeros(len(aves_df))
    sigdiff_cgs = [
        [[0, 0], [0, 1]],
        [[0, 1], [0, 2]],
    ]
    plot_rcgy(fig_fp, aves_df, xlabel="", ylabel=ylabel, group_labels=group_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, figsize=(24, 16))

    # fig_fp = "/home/phuong/data/phd-project/figures/fig_3b.png"
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/0--HEK-BL-intensity/results/y.csv"
    # group_labels = ["0", "1", "5", "10", "50"]
    # class_labels = ["Reporter Only", "Dense-RFP"]
    # palette = ["#34495E", "#8069EC"]
    # xlabel = r"$\mathdefault{Input\ Intensity\ (\mu W/mm^2)}$"
    # ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    # y_norm_df = calc_log2_ratio(y_csv_fp)
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/0--HEK-BL-intensity/results/y_norm.csv"
    # y_norm_df.to_csv(y_csv_fp, index=False)
    # plot_class_groups(fig_fp, y_csv_fp, class_labels, group_labels, xlabel, ylabel, palette=palette)

    # fig_fp = "/home/phuong/data/phd-project/figures/fig_3c.png"
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/1--HEK-FM_single/results/y.csv"
    # group_labels = ["0", "0.05", "0.1", "0.25", "0.5", "1"]
    # class_labels = ["Dense-RFP", "Sparse-RFP"]
    # palette = ["#8069EC", "#EA822C"]
    # xlabel = r"$\mathdefault{Input\ Pulse\ Freq.\ (Hz)}$"
    # ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    # y_norm_df = calc_log2_ratio(y_csv_fp)
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/1--HEK-FM_single/results/y_norm.csv"
    # y_norm_df.to_csv(y_csv_fp, index=False)
    # plot_class_groups(fig_fp, y_csv_fp, class_labels, group_labels, xlabel, ylabel, palette=palette)

    # fig_fp = "/home/phuong/data/phd-project/figures/fig_3e.png"
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/2--HEK-FM_dual/results/y.csv"
    # group_labels = ["None\nInput", "Sparse\nInput", "Dense\nInput"]
    # class_labels = ["Dense-YFP", "Sparse-RFP"]
    # palette = ["#8069EC", "#EA822C"]
    # xlabel = ""
    # ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    # y_norm_df = calc_log2_ratio(y_csv_fp)
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/2--HEK-FM_dual/results/y_norm.csv"
    # y_norm_df.to_csv(y_csv_fp, index=False)
    # plot_class_groups(fig_fp, y_csv_fp, class_labels, group_labels, xlabel, ylabel, palette=palette)

    # fig_fp = "/home/phuong/data/phd-project/figures/fig_4c.png"
    # y_csv_fp = "/home/phuong/data/phd-project/3--antigen/1--CAR-killing-assay/y.csv"
    # group_labels = [
    #     "None\nInput",
    #     "Sparse\nInput",
    #     "Dense\nInput",
    # ]
    # class_labels = ["Dense-CD19", "Sparse-PSMA"]
    # palette = ["#8069EC", "#EA822C"]
    # xlabel = ""
    # ylabel = "% Cytotoxicity"
    # plot_class_groups(fig_fp, y_csv_fp, class_labels, group_labels, xlabel, ylabel, palette=palette)


if __name__ == "__main__":
    main()
