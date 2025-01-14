from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ezplot.style import PALETTE, STYLE


def plot_groups(fig_fp, y_csv_fp, group_labels, ylabel, log2_yaxis=False, palette=PALETTE, figsize=(24, 16), rc_params=STYLE):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        y_df = pd.read_csv(y_csv_fp)
        sns.stripplot(
            ax=ax,
            data=y_df,
            x="group",
            y="response",
            hue="group",
            palette=palette,
            jitter=0.1,
            size=2 * rc_params["lines.markersize"],
            linewidth=rc_params["lines.markeredgewidth"],
            edgecolor=rc_params["lines.markeredgecolor"],
            alpha=0.8,
            zorder=2.1,
        )
        sns.pointplot(
            ax=ax,
            data=y_df,
            x="group",
            y="response",
            hue="group",
            errorbar=("se", 1.96),
            palette="dark:#212121",
            markers=".",
            err_kws={"linewidth": 10},
            linestyles=None,
            capsize=0.2,
            zorder=2.2,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels)
        if log2_yaxis:
            ax.set_yscale("log", base=2)
        ax.locator_params(axis="y", nbins=10)
        ax.get_legend().remove()
        fig.savefig(fig_fp)
    plt.close("all")


def plot_class_groups(fig_fp, y_csv_fp, class_labels, group_labels, xlabel, ylabel, palette=PALETTE, figsize=(24, 16), rc_params=STYLE):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        y_df = pd.read_csv(y_csv_fp)
        sns.stripplot(
            ax=ax,
            data=y_df,
            x="group",
            y="response",
            hue="class",
            palette=palette,
            jitter=0.1,
            dodge=True,
            size=2 * rc_params["lines.markersize"],
            linewidth=rc_params["lines.markeredgewidth"],
            edgecolor=rc_params["lines.markeredgecolor"],
            alpha=0.8,
            zorder=2.1,
        )
        sns.pointplot(
            ax=ax,
            data=y_df,
            x="group",
            y="response",
            hue="class",
            dodge=0.4,
            errorbar=("se", 1.96),
            palette="dark:#212121",
            markers=".",
            markersize=rc_params["lines.markersize"],
            err_kws={"linewidth": 10},
            linestyles=["" for c in class_labels],
            capsize=0.2,
            zorder=2.2,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, class_labels, loc="best")
        fig.savefig(fig_fp)
    plt.close("all")


def main():
    # fig_fp = "/home/phuong/data/phd-project/figures/fig_2j.png"
    # y_csv_fp = "/home/phuong/data/phd-project/1--biosensor/3--sparser/results/y.csv"
    # group_labels = [
    #     "None\nInput",
    #     "Sparse\nInput",
    #     "Dense\nInput",
    # ]
    # palette = ["#34495E", "#EA822C", "#8069EC"]
    # ylabel = r"$\mathbf{Ave\ \Delta F/F_{0}}$"
    # time_regimes = [[0, 60], [60, 120], [120, 155]]
    # ave_regimes_df = calc_ave_regimes(y_csv_fp, time_regimes)
    # y_csv_fp = "/home/phuong/data/phd-project/1--biosensor/3--sparser/results/ave_regimes.csv"
    # ave_regimes_df.to_csv(y_csv_fp, index=False)
    # ave_regimes_df = pd.read_csv(y_csv_fp)
    # ave_none = ave_regimes_df.loc[(ave_regimes_df["group"] == 0), ["response"]]
    # ave_sparse = ave_regimes_df.loc[(ave_regimes_df["group"] == 1), ["response"]]
    # ave_dense = ave_regimes_df.loc[(ave_regimes_df["group"] == 2), ["response"]]
    # print(sp.stats.ttest_ind(ave_none, ave_sparse))
    # print(sp.stats.ttest_ind(ave_sparse, ave_dense))
    # plot_groups(fig_fp, y_csv_fp, group_labels, ylabel=ylabel, palette=palette)

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

    fig_fp = "/home/phuong/data/phd-project/figures/fig_4c.png"
    y_csv_fp = "/home/phuong/data/phd-project/3--antigen/1--CAR-killing-assay/y.csv"
    group_labels = [
        "None\nInput",
        "Sparse\nInput",
        "Dense\nInput",
    ]
    class_labels = ["Dense-CD19", "Sparse-PSMA"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = ""
    ylabel = "% Cytotoxicity"
    plot_class_groups(fig_fp, y_csv_fp, class_labels, group_labels, xlabel, ylabel, palette=palette)


if __name__ == "__main__":
    main()
