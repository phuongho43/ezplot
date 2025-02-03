from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted

from ezplot.style import PALETTE, STYLE


def plot_crty(fig_fp, crty_df, class_labels, xlabel="Time", ylabel="Response", tu_df=None, figsize=(24, 16), leg_loc="best", palette=PALETTE, rc_params=STYLE):
    """Plot timeseries data.

    Args:
        fig_fp (str): absolute path for saving generated figure
        crty_df (DataFrame): DataFrame with columns [c, r, t, y] describing biological response (y) over time (t) for each class (c) and repeat (r)
        class_labels (list): list of labels for each timeseries
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        tu_df (DataFrame): DataFrame with columns [t, u] describing stimuli input (u) over time (t)
        figsize (tuple): figure dimensions (width, height)
        palette (list): list of hex color codes for each group
        rc_params (dict): matplotlib style settings
    """
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        handles = []
        lw = rc_params["lines.linewidth"]
        if tu_df is not None:
            dt = tu_df["t"].diff().mean()
            for t in tu_df.loc[(tu_df["u"] > 0), "t"]:
                ax.axvspan(t, t + dt, color="#648FFF", alpha=0.8, lw=0)
            handles.append(mpl.lines.Line2D([], [], color="#648FFF", lw=lw, alpha=0.8, solid_capstyle="projecting"))
            class_labels.insert(0, "Input")
        lsizes = np.linspace(18, 6, crty_df["c"].nunique(), endpoint=True).tolist()
        sns.lineplot(ax=ax, data=crty_df, x="t", y="y", hue="c", size="c", sizes=lsizes, errorbar=("se", 1.96), palette=palette)
        for i in range(crty_df["c"].nunique()):
            handles.append(mpl.lines.Line2D([], [], color=palette[i], lw=lw))
        ax.legend(handles, class_labels, loc=leg_loc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def convert_tu(tt, u_ta_tb_csv_fp):
    u_ta_tb_df = pd.read_csv(u_ta_tb_csv_fp)
    u_ta = np.round(u_ta_tb_df["ta"].values, 1)
    u_tb = np.round(u_ta_tb_df["tb"].values, 1)
    tt = np.arange(tt[0], tt[-1], 0.1)
    tt = np.round(tt, 1)
    uu = np.zeros_like(tt)
    for ta, tb in zip(u_ta, u_tb):
        if ta > tt[-1]:
            continue
        if tb > tt[-1]:
            tb = tt[-1]
        ia = np.where(tt == ta)[0][0]
        ib = np.where(tt == tb)[0][0]
        uu[ia:ib] = 1.0
    tu_df = pd.DataFrame({"t": tt, "u": uu})
    return tu_df


def analyze_crty_tu_data(class_dps):
    crty_df = []
    crtu_df = []
    for c, class_dp in enumerate([Path(dp) for dp in class_dps]):
        rt_ave = []
        rty_df = []
        for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(class_dp.glob("*")) if rep_dp.is_dir()]):
            ty_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(ty_csv_fp)
            F0 = ty_df["y"].iloc[:5].mean()
            dF = ty_df["y"] - F0
            ty_df["y"] = dF / F0
            ty_df["c"] = np.ones(len(ty_df), dtype=int) * c
            ty_df["r"] = np.ones(len(ty_df), dtype=int) * r
            rty_df.append(ty_df)
            rt_ave.append(ty_df["t"].values)
            u_ta_tb_csv_fp = rep_dp / "u.csv"
            tu_df = convert_tu(ty_df["t"].values, u_ta_tb_csv_fp)
            tu_df["c"] = np.ones(len(tu_df), dtype=int) * c
            tu_df["r"] = np.ones(len(tu_df), dtype=int) * r
            crtu_df.append(tu_df)
        rt_ave = np.array(rt_ave).mean(axis=0)
        for df in rty_df:
            df["t"] = rt_ave
        crty_df.extend(rty_df)
    crty_df = pd.concat(crty_df)
    crtu_df = pd.concat(crtu_df)
    tu_df = crtu_df.groupby("t", as_index=False)["u"].mean()
    return crty_df, tu_df


def main():
    ## Plot Each Rep ##
    expt_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/0--LOVfast-BL20uW/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/1--LOVfast-BL200uW/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/1--V416I/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/0--iLIDslow-13AA/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/1--iLIDslow-20AA/",
        "/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/",
    ]
    for expt_dp in [Path(expt_dp) for expt_dp in expt_dps]:
        for rep_dp in [dp for dp in natsorted(expt_dp.glob("*")) if dp.is_dir()]:
            y_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(y_csv_fp)
            F0 = ty_df["y"].iloc[:5].mean()
            dF = ty_df["y"] - F0
            ty_df["y"] = dF / F0
            ty_df["c"] = np.zeros(len(ty_df), dtype=int)
            class_labels = ["y"]
            xlabel = "Time (s)"
            ylabel = r"$\mathbf{\Delta F/F_{0}}$"
            palette = ["#8069EC"]
            fig_fp = rep_dp / "results" / "y.png"
            plot_crty(fig_fp, ty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=None, palette=palette)

    ## Figure 2C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2c.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/",
    ]
    crty_df, tu_df = analyze_crty_tu_data(class_dps)
    class_labels = [
        "ddFP",
        "LOVfast",
        "LOVslow",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, palette=palette)

    ## Figure S1A ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1a.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/0--LOVfast-BL20uW/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/1--LOVfast-BL200uW/",
    ]
    crty_df, tu_df = analyze_crty_tu_data(class_dps)
    crty_df = crty_df.loc[(crty_df["t"] < 181)]
    tu_df = tu_df.loc[(tu_df["t"] < 181)]
    class_labels = [
        r"$\mathdefault{20\ \mu W/mm^2}$",
        r"$\mathdefault{200\ \mu W/mm^2}$",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#2ECC71", "#EA822C"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, palette=palette)

    ## Figure 2F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2f.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/1--V416I/",
    ]
    crty_df, tu_df = analyze_crty_tu_data(class_dps)
    class_labels = [
        "ddFP",
        "iLIDfast",
        "iLIDslow",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, palette=palette)

    ## Figure S1C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1c.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/4--linker/0--iLIDslow-13AA/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/1--iLIDslow-20AA/",
    ]
    crty_df, tu_df = analyze_crty_tu_data(class_dps)
    crty_df = crty_df.loc[(crty_df["t"] < 181)]
    tu_df = tu_df.loc[(tu_df["t"] < 181)]
    class_labels = [
        "13 AA Linker",
        "20 AA Linker",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#2ECC71", "#EA822C"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, palette=palette)

    # Figure 2I ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2i.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/",
    ]
    crty_df, tu_df = analyze_crty_tu_data(class_dps)
    crty_df = crty_df.loc[(crty_df["t"] < 155)]
    tu_df = tu_df.loc[(tu_df["t"] < 155)]
    class_labels = [
        "Sparse-ddFP",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#EA822C"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, leg_loc="lower left", palette=palette)

    ## Figure 4F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4f.png"
    cty_df = pd.read_csv("/home/phuong/data/phd-project/4--antigen/2--CAR-bilateral-tumor/y.csv")
    class_labels = [
        "Dense-CD19 (Dense Input)",
        "Sparse-PSMA (Dense Input)",
        "Dense-CD19 (Sparse Input)",
        "Sparse-PSMA (Sparse Input)",
    ]
    tu_df = None
    xlabel = "Time (d)"
    ylabel = "Total Flux (p/s)"
    palette = ["#8069EC", "#a67750", "#6c6296", "#EA822C"]
    plot_crty(fig_fp, cty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, figsize=(24, 16), palette=palette)


if __name__ == "__main__":
    main()
