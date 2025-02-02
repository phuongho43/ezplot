from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted

from ezplot.process import convert_u_ta_tb
from ezplot.style import PALETTE, STYLE


def plot_dynamics(fig_fp, y_df, group_labels, xlabel="Time", ylabel="Response", u_df=None, figsize=(24, 16), leg_loc="best", palette=PALETTE, rc_params=STYLE):
    """Plot timeseries data.

    Args:
        fig_fp (str): absolute path for saving generated figure
        y_df (DataFrame): DataFrame with columns [t, y, g] describing response output (y) over time (t) for each group (g)
        group_labels (list): list of expt group labels for each timeseries
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        u_df (DataFrame): DataFrame with columns [t, u] describing stimuli input over time
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
        if u_df is not None:
            dt = u_df["t"].diff().mean()
            for t in u_df.loc[(u_df["u"] > 0), "t"]:
                ax.axvspan(t, t + dt, color="#648FFF", alpha=0.8, lw=0)
            handles.append(mpl.lines.Line2D([], [], color="#648FFF", lw=lw, alpha=0.8, solid_capstyle="projecting"))
            group_labels.insert(0, "Input")
        lsizes = np.linspace(18, 6, y_df["g"].nunique(), endpoint=True).tolist()
        sns.lineplot(ax=ax, data=y_df, x="t", y="y", hue="g", size="g", sizes=lsizes, errorbar=("se", 1.96), palette=palette)
        for i in range(y_df["g"].nunique()):
            handles.append(mpl.lines.Line2D([], [], color=palette[i], lw=lw))
        ax.legend(handles, group_labels, loc=leg_loc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def analyze_biosensor_data(group_dps):
    grty_df = []
    grtu_df = []
    for g, group_dp in enumerate([Path(group_dp) for group_dp in group_dps]):
        rt_ave = []
        rty_df = []
        for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(group_dp.glob("*")) if rep_dp.is_dir()]):
            ty_csv_fp = rep_dp / "results" / "y.csv"
            ty_df = pd.read_csv(ty_csv_fp)
            F0 = ty_df["y"].iloc[:5].mean()
            dF = ty_df["y"] - F0
            ty_df["y"] = dF / F0
            ty_df["g"] = np.ones(len(ty_df), dtype=int) * g
            ty_df["r"] = np.ones(len(ty_df), dtype=int) * r
            rty_df.append(ty_df)
            rt_ave.append(ty_df["t"].values)
            tu_csv_fp = rep_dp / "u.csv"
            tu_df = convert_u_ta_tb(ty_df["t"].values, tu_csv_fp)
            tu_df["g"] = np.ones(len(tu_df), dtype=int) * g
            tu_df["r"] = np.ones(len(tu_df), dtype=int) * r
            grtu_df.append(tu_df)
        rt_ave = np.array(rt_ave).mean(axis=0)
        for df in rty_df:
            df["t"] = rt_ave
        grty_df.extend(rty_df)
    grty_df = pd.concat(grty_df)
    grtu_df = pd.concat(grtu_df)
    grtu_df = grtu_df.groupby("t", as_index=False)["u"].mean()
    return grty_df, grtu_df


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
            ty_df["g"] = np.zeros(len(ty_df), dtype=int)
            group_labels = ["y"]
            xlabel = "Time (s)"
            ylabel = r"$\mathbf{\Delta F/F_{0}}$"
            palette = ["#8069EC"]
            fig_fp = rep_dp / "results" / "y.png"
            plot_dynamics(fig_fp, ty_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=None, palette=palette)

    ## Figure 2C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2c.png"
    group_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/",
    ]
    grty_df, grtu_df = analyze_biosensor_data(group_dps)
    group_labels = [
        "ddFP",
        "LOVfast",
        "LOVslow",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    plot_dynamics(fig_fp, grty_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=grtu_df, palette=palette)

    ## Figure S1A ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1a.png"
    group_dps = [
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/0--LOVfast-BL20uW/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/1--LOVfast-BL200uW/",
    ]
    grty_df, grtu_df = analyze_biosensor_data(group_dps)
    grty_df = grty_df.loc[(grty_df["t"] < 181)]
    grtu_df = grtu_df.loc[(grtu_df["t"] < 181)]
    group_labels = [
        r"$20\ \mu W/mm^2$",
        r"$200\ \mu W/mm^2$",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#2ECC71", "#EA822C"]
    plot_dynamics(fig_fp, grty_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=grtu_df, palette=palette)

    ## Figure 2F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2f.png"
    group_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/1--V416I/",
    ]
    grty_df, grtu_df = analyze_biosensor_data(group_dps)
    group_labels = [
        "ddFP",
        "iLIDfast",
        "iLIDslow",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    plot_dynamics(fig_fp, grty_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=grtu_df, palette=palette)

    ## Figure S1C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1c.png"
    group_dps = [
        "/home/phuong/data/phd-project/1--biosensor/4--linker/0--iLIDslow-13AA/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/1--iLIDslow-20AA/",
    ]
    grty_df, grtu_df = analyze_biosensor_data(group_dps)
    grty_df = grty_df.loc[(grty_df["t"] < 181)]
    grtu_df = grtu_df.loc[(grtu_df["t"] < 181)]
    group_labels = [
        "13 AA Linker",
        "20 AA Linker",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#2ECC71", "#EA822C"]
    plot_dynamics(fig_fp, grty_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=grtu_df, palette=palette)

    # Figure 2I ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2i.png"
    group_dps = [
        "/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/",
    ]
    grty_df, grtu_df = analyze_biosensor_data(group_dps)
    grty_df = grty_df.loc[(grty_df["t"] < 155)]
    grtu_df = grtu_df.loc[(grtu_df["t"] < 155)]
    group_labels = [
        "Sparse-ddFP",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#EA822C"]
    plot_dynamics(fig_fp, grty_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=grtu_df, leg_loc="lower left", palette=palette)

    ## Figure 4F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4f.png"
    y_df = pd.read_csv("/home/phuong/data/phd-project/4--antigen/2--CAR-bilateral-tumor/y.csv")
    group_labels = [
        "Dense-CD19 (Dense Input)",
        "Sparse-PSMA (Dense Input)",
        "Dense-CD19 (Sparse Input)",
        "Sparse-PSMA (Sparse Input)",
    ]
    u_df = None
    xlabel = "Time (d)"
    ylabel = "Total Flux (p/s)"
    palette = ["#8069EC", "#a67750", "#6c6296", "#EA822C"]
    plot_dynamics(fig_fp, y_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=u_df, figsize=(24, 16), palette=palette)


if __name__ == "__main__":
    main()
