from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from natsort import natsorted

from ezplot.plot_numericals import calc_dF_F0
from ezplot.style import PALETTE, STYLE


def draw_sigdiff(ax, df, cg1, cg2, sy):
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
    y = max(cg1_vals.max(), cg2_vals.max(), cg1_err_top, cg2_err_top) + dy * sy
    pval = sp.stats.ttest_ind(cg1_vals, cg2_vals).pvalue
    va = "center"
    fontsize = STYLE["font.size"]
    if pval <= 0.001:
        fontsize *= 1.5
        text = "***"
    elif pval <= 0.01:
        fontsize *= 1.5
        text = "**"
    elif pval <= 0.05:
        fontsize *= 1.5
        text = "*"
    else:
        text = "ns"
        va = "bottom"
    bar_x = [x1, x1, x2, x2]
    bar_y = [y, y + dy / 2, y + dy / 2, y]
    ax.plot(bar_x, bar_y, color="#212121", lw=STYLE["axes.linewidth"], zorder=10.1)
    ax.text(xm, y + dy / 2, text, fontsize=fontsize, ha="center", va=va, zorder=10.2)


def plot_cgry(fig_fp, cgry_df, xlabel, ylabel, group_labels, class_labels=None, signi_bar=None, figsize=(24, 16), leg_loc="best", palette=PALETTE, rc_params=STYLE):
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
            data=cgry_df,
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
            data=cgry_df,
            x="group",
            y="response",
            hue=hue,
            dodge=dodge,
            errorbar=("se", 1.96),
            palette="dark:#212121",
            markers=".",
            markersize=rc_params["lines.markersize"],
            err_kws={"linewidth": rc_params["axes.linewidth"]},
            ls="",
            capsize=0.2,
            zorder=2.2,
        )
        if signi_bar is not None:
            for cg1, cg2, sy in signi_bar:
                draw_sigdiff(ax=ax, df=cgry_df, cg1=cg1, cg2=cg2, sy=sy)
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


def calc_log2_norm(cgry_df):
    for c in cgry_df["class"].unique():
        for r in cgry_df["repeat"].unique():
            y = cgry_df.loc[(cgry_df["class"] == c) & (cgry_df["repeat"] == r), "response"]
            cgry_df.loc[(cgry_df["class"] == c) & (cgry_df["repeat"] == r), "response"] = np.log2(y / y.iloc[0])
    return cgry_df


def prep_cgry_data(expt_dp):
    df = []
    for c, class_dp in enumerate([dp for dp in natsorted(Path(expt_dp).glob("*")) if dp.is_dir()]):
        for g, group_dp in enumerate([dp for dp in natsorted(class_dp.glob("*")) if dp.is_dir()]):
            for r, rep_dp in enumerate([dp for dp in natsorted(group_dp.glob("*")) if dp.is_dir()]):
                y_csv_fp = rep_dp / "results" / "y.csv"
                y_df = pd.read_csv(y_csv_fp)
                y = y_df["y"].mean()
                df.append({"class": c, "group": g, "repeat": r, "response": y})
    df = pd.DataFrame(df)
    return df


def main():
    ## Figure S1B ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1b.png"
    expt_dp = "/home/phuong/data/phd-project/1--biosensor/2--intensity/"
    data = []
    for g, group_dp in enumerate([group_dp for group_dp in natsorted(Path(expt_dp).glob("*")) if group_dp.is_dir()]):
        for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(group_dp.glob("*")) if rep_dp.is_dir()]):
            ty_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(ty_csv_fp)
            ty_df = calc_dF_F0(ty_df)
            y = ty_df["y"].min()
            cgry = {"class": 0, "group": g, "repeat": r, "response": y}
            data.append(cgry)
    df = pd.DataFrame(data)
    signi_bar = [
        [[0, 0], [0, 1], 1],
    ]
    group_labels = ["20", "200"]
    palette = ["#2ECC71", "#EA822C"]
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    xlabel = r"$\mathdefault{Input\ Intensity\ (\mu W/mm^2)}$"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, signi_bar=signi_bar, palette=palette)

    ## Figure S1D ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1d.png"
    expt_dp = "/home/phuong/data/phd-project/1--biosensor/4--linker/"
    data = []
    for g, group_dp in enumerate([group_dp for group_dp in natsorted(Path(expt_dp).glob("*")) if group_dp.is_dir()]):
        for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(group_dp.glob("*")) if rep_dp.is_dir()]):
            ty_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(ty_csv_fp)
            ty_df = calc_dF_F0(ty_df)
            y = ty_df["y"].max()
            cgry = {"class": 0, "group": g, "repeat": r, "response": y}
            data.append(cgry)
    df = pd.DataFrame(data)
    signi_bar = [
        [[0, 0], [0, 1], 1],
    ]
    group_labels = [
        "13",
        "20",
    ]
    palette = ["#2ECC71", "#EA822C"]
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    xlabel = "Linker Length (AA)"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, signi_bar=signi_bar, palette=palette)

    ## Figure 2J ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2j.png"
    expt_dp = "/home/phuong/data/phd-project/1--biosensor/5--decoder/"
    tspans = [[0, 60], [60, 120], [120, 155]]
    data = []
    for c, class_dp in enumerate([class_dp for class_dp in natsorted(Path(expt_dp).glob("*")) if class_dp.is_dir()]):
        for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(class_dp.glob("*")) if rep_dp.is_dir()]):
            ty_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(ty_csv_fp)
            ty_df = calc_dF_F0(ty_df)
            for g, [t1, t2] in enumerate(tspans):
                y = ty_df.loc[(ty_df["t"] >= t1) & (ty_df["t"] < t2)]["y"].mean()
                cgry = {"class": c, "group": g, "repeat": r, "response": y}
                data.append(cgry)
    df = pd.DataFrame(data)
    signi_bar = [
        [[0, 0], [0, 1], 1],
        [[0, 1], [0, 2], 1],
    ]
    group_labels = [
        "None\nInput",
        "Sparse\nInput",
        "Dense\nInput",
    ]
    palette = ["#34495E", "#EA822C", "#8069EC"]
    ylabel = r"$\mathbf{Ave\ \Delta F/F_{0}}$"
    plot_cgry(fig_fp, df, xlabel="", ylabel=ylabel, group_labels=group_labels, signi_bar=signi_bar, palette=palette)

    ## Figure 4B ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4b.png"
    expt_dp = "/home/phuong/data/phd-project/3--expression/0--293T-intensity/"
    df = prep_cgry_data(expt_dp)
    df = calc_log2_norm(df)
    signi_bar = [
        [[1, 1], [1, 2], 1],
        [[1, 2], [1, 3], 1],
        [[1, 3], [1, 4], 1],
    ]
    group_labels = ["0", "1", "5", "10", "50"]
    class_labels = ["Reporter Only", "Dense-RFP"]
    palette = ["#34495E", "#8069EC"]
    xlabel = r"$\mathdefault{Input\ Intensity\ (\mu W/mm^2)}$"
    ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    leg_loc = "upper left"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, signi_bar=signi_bar, palette=palette, leg_loc=leg_loc)

    ## Figure 4C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4c.png"
    expt_dp = "/home/phuong/data/phd-project/3--expression/1--293T-FM-single/"
    df = prep_cgry_data(expt_dp)
    df = calc_log2_norm(df)
    signi_bar = [
        [[0, 3], [1, 3], 1],
        [[0, 5], [1, 5], 1],
    ]
    group_labels = ["0", "0.05", "0.1", "0.25", "0.5", "1"]
    class_labels = ["Dense-RFP", "Sparse-RFP"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = r"$\mathdefault{Input\ Pulse\ Freq.\ (Hz)}$"
    ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    leg_loc = "upper left"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, signi_bar=signi_bar, palette=palette, leg_loc=leg_loc)

    ## Figure 4D ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4d.png"
    expt_dp = "/home/phuong/data/phd-project/3--expression/2--293T-iLID-vs-LOV/"
    df = prep_cgry_data(expt_dp)
    df = calc_log2_norm(df)
    signi_bar = [
        [[1, 0], [1, 1], 1],
        [[1, 0], [1, 2], 5],
        [[0, 1], [0, 2], 1],
    ]
    group_labels = ["None\nInput", "Sparse\nInput", "Dense\nInput"]
    class_labels = ["LOVfast", "iLIDslow"]
    palette = ["#2ECC71", "#D143A4"]
    xlabel = ""
    ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    leg_loc = "lower left"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, signi_bar=signi_bar, palette=palette, leg_loc=leg_loc)

    ## Figure 4E ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4e.png"
    expt_dp = "/home/phuong/data/phd-project/3--expression/3--293T-FM-dual/"
    df = prep_cgry_data(expt_dp)
    df = calc_log2_norm(df)
    signi_bar = [
        [[0, 1], [1, 1], 1],
        [[0, 2], [1, 2], 1],
    ]
    group_labels = ["None\nInput", "Sparse\nInput", "Dense\nInput"]
    class_labels = ["Dense-YFP", "Sparse-RFP"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = ""
    ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    leg_loc = "upper left"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, signi_bar=signi_bar, palette=palette, leg_loc=leg_loc)

    ## Figure S3 ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s3.png"
    expt_dp = "/home/phuong/data/phd-project/3--expression/4--K562-FM-single/"
    df = prep_cgry_data(expt_dp)
    df = calc_log2_norm(df)
    signi_bar = [
        [[0, 3], [1, 3], 1],
        [[0, 5], [1, 5], 1],
    ]
    group_labels = ["0", "0.05", "0.1", "0.25", "0.5", "1"]
    class_labels = ["Dense-RFP", "Sparse-RFP"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = r"$\mathdefault{Input\ Pulse\ Freq.\ (Hz)}$"
    ylabel = r"$\mathdefault{Log_2\ Norm.\ Output}$"
    leg_loc = "upper left"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, signi_bar=signi_bar, palette=palette, leg_loc=leg_loc)

    ## Figure 5C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_5c.png"
    y_csv_fp = "/home/phuong/data/phd-project/4--antigen/1--CAR-killing-assay/y.csv"
    df = pd.read_csv(y_csv_fp)
    signi_bar = [
        [[0, 1], [1, 1], 1],
        [[0, 2], [1, 2], 1],
    ]
    group_labels = [
        "None\nInput",
        "Sparse\nInput",
        "Dense\nInput",
    ]
    class_labels = ["Dense-CD19", "Sparse-PSMA"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = ""
    ylabel = "% Cytotoxicity"
    leg_loc = "upper left"
    plot_cgry(fig_fp, df, xlabel=xlabel, ylabel=ylabel, group_labels=group_labels, class_labels=class_labels, signi_bar=signi_bar, palette=palette, leg_loc=leg_loc)

    print("\a")


if __name__ == "__main__":
    main()
