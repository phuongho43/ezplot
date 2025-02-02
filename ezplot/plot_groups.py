from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from natsort import natsorted

from ezplot.style import PALETTE, STYLE


def draw_sigdiff(ax, df, cg1, cg2):
    X = []
    for line in ax.lines:
        Xi = line.get_xdata()
        n_groups = df["group"].nunique()
        if len(Xi) == n_groups and not np.any(np.isnan(Xi)):
            X.append(Xi)
    X = np.array(X)
    x1 = X[cg1[0], cg1[1]]
    x2 = X[cg2[0], cg2[1]]
    xm = (x1 + x2) / 2
    cg1_vals = df.loc[(df["class"] == cg1[0]) & (df["group"] == cg1[1]), "response"]
    cg2_vals = df.loc[(df["class"] == cg2[0]) & (df["group"] == cg2[1]), "response"]
    dy = (df["response"].max() - df["response"].min()) / 20
    cg1_err_top = cg1_vals.mean() + 1.96 * cg1_vals.sem()
    cg2_err_top = cg2_vals.mean() + 1.96 * cg2_vals.sem()
    y = max(cg1_vals.max(), cg2_vals.max(), cg1_err_top, cg2_err_top) + dy
    pval = sp.stats.ttest_ind(cg1_vals, cg2_vals).pvalue
    va = "center"
    if pval <= 0.001:
        text = "***"
    elif pval <= 0.01:
        text = "**"
    elif pval <= 0.05:
        text = "*"
    else:
        text = "ns"
        va = "bottom"
    bar_x = [x1, x1, x2, x2]
    bar_y = [y, y + dy / 2, y + dy / 2, y]
    ax.plot(bar_x, bar_y, color="#212121", lw=8, zorder=10.1)
    ax.text(xm, y + dy / 2, text, ha="center", va=va, zorder=10.2)


def plot_cgry(fig_fp, rcgy_df, xlabel, ylabel, group_labels, class_labels=None, sigdiff_cgs=None, palette=PALETTE, rc_params=STYLE, figsize=(24, 16), leg_loc="best"):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        if class_labels is not None:
            hue = "class"
            dodge = 0.4
        else:
            hue = "group"
            dodge = False
        sns.stripplot(
            ax=ax,
            data=rcgy_df,
            x="group",
            y="response",
            hue=hue,
            dodge=dodge,
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
            dodge=dodge,
            errorbar=("se", 1.96),
            palette="dark:#212121",
            markers=".",
            markersize=rc_params["lines.markersize"],
            err_kws={"linewidth": 10},
            ls="",
            capsize=0.2,
            zorder=2.2,
        )
        if sigdiff_cgs is not None:
            for cg1, cg2 in sigdiff_cgs:
                draw_sigdiff(ax=ax, df=rcgy_df, cg1=cg1, cg2=cg2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels)
        if class_labels is not None:
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, class_labels, loc=leg_loc).set_zorder(20)
        else:
            ax.get_legend().remove()
        fig.savefig(fig_fp)
    plt.close("all")


def main():
    ## Figure 2J ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2j.png"
    expt_dp = "/home/phuong/data/phd-project/1--biosensor/5--decoder/"
    tspans = [[0, 60], [60, 120], [120, 155]]
    data = []
    for c, class_dp in enumerate([class_dp for class_dp in natsorted(Path(expt_dp).glob("*")) if class_dp.is_dir()]):
        for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(class_dp.glob("*")) if rep_dp.is_dir()]):
            ty_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(ty_csv_fp)
            F0 = ty_df["y"].iloc[:5].mean()
            dF = ty_df["y"] - F0
            ty_df["y"] = dF / F0
            for g, [t1, t2] in enumerate(tspans):
                y = ty_df.loc[(ty_df["t"] >= t1) & (ty_df["t"] < t2)]["y"].mean()
                cgry = {"class": c, "group": g, "repeat": r, "response": y}
                data.append(cgry)
    df = pd.DataFrame(data)
    sigdiff_cgs = [
        [[0, 0], [0, 1]],
        [[0, 1], [0, 2]],
    ]
    group_labels = [
        "None\nInput",
        "Sparse\nInput",
        "Dense\nInput",
    ]
    palette = ["#34495E", "#EA822C", "#8069EC"]
    ylabel = r"$\mathbf{Ave\ \Delta F/F_{0}}$"
    plot_cgry(fig_fp, df, xlabel="", ylabel=ylabel, group_labels=group_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, figsize=(24, 16))

    ## Figure S1B ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1b.png"
    expt_dp = "/home/phuong/data/phd-project/1--biosensor/2--intensity/"
    data = []
    for g, group_dp in enumerate([group_dp for group_dp in natsorted(Path(expt_dp).glob("*")) if group_dp.is_dir()]):
        for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(group_dp.glob("*")) if rep_dp.is_dir()]):
            ty_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(ty_csv_fp)
            F0 = ty_df["y"].iloc[:5].mean()
            dF = ty_df["y"] - F0
            ty_df["y"] = dF / F0
            y = ty_df["y"].min()
            cgry = {"class": 0, "group": g, "repeat": r, "response": y}
            data.append(cgry)
    df = pd.DataFrame(data)
    sigdiff_cgs = [
        [[0, 0], [0, 1]],
        [[0, 0], [0, 2]],
    ]
    group_labels = ["20", "200", "2000"]
    palette = ["#2ECC71", "#F1C40F", "#EA822C"]
    ylabel = r"$\mathbf{Ave\ \Delta F/F_{0}}$"
    xlabel = r"$\mathdefault{Input\ Intensity\ (\mu W/mm^2)}$"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, figsize=(24, 16))

    ## Figure S1D ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1d.png"
    expt_dp = "/home/phuong/data/phd-project/1--biosensor/4--linker/"
    data = []
    for c, class_dp in enumerate([class_dp for class_dp in natsorted(Path(expt_dp).glob("*")) if class_dp.is_dir()]):
        for g, group_dp in enumerate([group_dp for group_dp in natsorted(class_dp.glob("*")) if group_dp.is_dir()]):
            for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(group_dp.glob("*")) if rep_dp.is_dir()]):
                ty_csv_fp = rep_dp / "results" / "y.csv"
                ty_df = pd.read_csv(ty_csv_fp)
                F0 = ty_df["y"].iloc[:5].mean()
                dF = ty_df["y"] - F0
                ty_df["y"] = dF / F0
                y = ty_df["y"].max()
                cgry = {"class": c, "group": g, "repeat": r, "response": y}
                data.append(cgry)
    df = pd.DataFrame(data)
    sigdiff_cgs = [
        [[0, 0], [1, 0]],
        [[0, 1], [1, 1]],
    ]
    class_labels = ["13 AA Linker", "20 AA Linker"]
    group_labels = [
        "iLIDfast",
        "iLIDslow",
    ]
    palette = ["#2ECC71", "#EA822C"]
    ylabel = r"$\mathbf{Ave\ \Delta F/F_{0}}$"
    xlabel = ""
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, leg_loc="upper right", figsize=(24, 16))

    # ## Figure 3B ##
    # fig_fp = "/home/phuong/data/phd-project/figures/fig_3b.png"
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/0--HEK-BL-intensity/results/y.csv"
    # rcgy_df = pd.read_csv(y_csv_fp)
    # norm_df = calc_log2_norm(rcgy_df)
    # sigdiff_cgs = [
    #     [[1, 1], [1, 2]],
    #     [[1, 2], [1, 3]],
    #     [[1, 3], [1, 4]],
    # ]
    # group_labels = ["0", "1", "5", "10", "50"]
    # class_labels = ["Reporter Only", "Dense-RFP"]
    # palette = ["#34495E", "#8069EC"]
    # xlabel = r"$\mathdefault{Input\ Intensity\ (\mu W/mm^2)}$"
    # ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    # leg_loc = "upper left"
    # plot_cgry(fig_fp, norm_df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, figsize=(24, 16), leg_loc=leg_loc)

    # ## Figure 3C ##
    # fig_fp = "/home/phuong/data/phd-project/figures/fig_3c.png"
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/1--HEK-FM_single/results/y.csv"
    # rcgy_df = pd.read_csv(y_csv_fp)
    # norm_df = calc_log2_norm(rcgy_df)
    # sigdiff_cgs = [
    #     [[0, 3], [1, 3]],
    #     [[0, 5], [1, 5]],
    # ]
    # group_labels = ["0", "0.05", "0.1", "0.25", "0.5", "1"]
    # class_labels = ["Dense-RFP", "Sparse-RFP"]
    # palette = ["#8069EC", "#EA822C"]
    # xlabel = r"$\mathdefault{Input\ Pulse\ Freq.\ (Hz)}$"
    # ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    # leg_loc = "upper left"
    # plot_cgry(fig_fp, norm_df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, figsize=(24, 16), leg_loc=leg_loc)

    # ## Figure 3E ##
    # fig_fp = "/home/phuong/data/phd-project/figures/fig_3e.png"
    # y_csv_fp = "/home/phuong/data/phd-project/2--expression/2--HEK-FM_dual/results/y.csv"
    # rcgy_df = pd.read_csv(y_csv_fp)
    # norm_df = calc_log2_norm(rcgy_df)
    # sigdiff_cgs = [
    #     [[0, 1], [1, 1]],
    #     [[0, 2], [1, 2]],
    # ]
    # group_labels = ["None\nInput", "Sparse\nInput", "Dense\nInput"]
    # class_labels = ["Dense-YFP", "Sparse-RFP"]
    # palette = ["#8069EC", "#EA822C"]
    # xlabel = ""
    # ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    # leg_loc = "upper left"
    # plot_cgry(fig_fp, norm_df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, figsize=(24, 16), leg_loc=leg_loc)

    # ## Figure 4C ##
    # fig_fp = "/home/phuong/data/phd-project/figures/fig_4c.png"
    # y_csv_fp = "/home/phuong/data/phd-project/3--antigen/1--CAR-killing-assay/y.csv"
    # rcgy_df = pd.read_csv(y_csv_fp)
    # sigdiff_cgs = [
    #     [[0, 1], [1, 1]],
    #     [[0, 2], [1, 2]],
    # ]
    # group_labels = [
    #     "None\nInput",
    #     "Sparse\nInput",
    #     "Dense\nInput",
    # ]
    # class_labels = ["Dense-CD19", "Sparse-PSMA"]
    # palette = ["#8069EC", "#EA822C"]
    # xlabel = ""
    # ylabel = "% Cytotoxicity"
    # leg_loc = "upper left"
    # plot_cgry(fig_fp, rcgy_df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, sigdiff_cgs=sigdiff_cgs, palette=palette, figsize=(24, 16), leg_loc=leg_loc)


if __name__ == "__main__":
    main()
