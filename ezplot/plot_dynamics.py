from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ezplot.process import combine_csv
from ezplot.style import PALETTE, STYLE


def plot_dynamics(fig_fp, y_df, group_labels, xlabel="Time", ylabel="Response", u_df=None, figsize=(24, 16), palette=PALETTE, rc_params=STYLE):
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
        sns.lineplot(ax=ax, data=y_df, x="t", y="y", hue="g", size="g", sizes=(18, 6), errorbar=("se", 1.96), palette=palette)
        for i in range(y_df["g"].nunique()):
            handles.append(mpl.lines.Line2D([], [], color=palette[i], lw=lw))
        ax.legend(handles, group_labels, loc="best")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def main():
    ## Figure 2C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2c.png"
    y_csv_fps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/y.csv",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/results/y.csv",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/results/y.csv",
    ]
    y_df = combine_csv(y_csv_fps, "g")
    group_labels = [
        "ddFP",
        "LOVfast",
        "LOVslow",
    ]
    u_csv_fps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/u.csv",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/results/u.csv",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/results/u.csv",
    ]
    u_df = combine_csv(u_csv_fps, "g")
    u_df = u_df.groupby("t", as_index=False)["u"].mean()
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    plot_dynamics(fig_fp, y_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=u_df, figsize=(24, 16), palette=palette)

    ## Figure 2F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2f.png"
    y_csv_fps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/y.csv",
        "/home/phuong/data/phd-project/1--biosensor/3--LID/0--I427V/results/y.csv",
        "/home/phuong/data/phd-project/1--biosensor/3--LID/1--V416I/results/y.csv",
    ]
    y_df = combine_csv(y_csv_fps, "g")
    group_labels = [
        "ddFP",
        "LIDfast",
        "LIDslow",
    ]
    u_csv_fps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/u.csv",
        "/home/phuong/data/phd-project/1--biosensor/3--LID/0--I427V/results/u.csv",
        "/home/phuong/data/phd-project/1--biosensor/3--LID/1--V416I/results/u.csv",
    ]
    u_df = combine_csv(u_csv_fps, "g")
    u_df = u_df.groupby("t", as_index=False)["u"].mean()
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    plot_dynamics(fig_fp, y_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=u_df, figsize=(24, 16), palette=palette)

    ## Figure 2I ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2i.png"
    y_csv_fps = ["/home/phuong/data/phd-project/1--biosensor/5--sparser/results/y.csv"]
    y_df = combine_csv(y_csv_fps, "g")
    group_labels = [
        "Sparse\nDecoder",
    ]
    u_csv_fps = ["/home/phuong/data/phd-project/1--biosensor/5--sparser/results/u.csv"]
    u_df = combine_csv(u_csv_fps, "g")
    u_df = u_df.groupby("t", as_index=False)["u"].mean()
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#EA822C"]
    plot_dynamics(fig_fp, y_df, group_labels, xlabel=xlabel, ylabel=ylabel, u_df=u_df, figsize=(24, 16), palette=palette)

    ## Figure 4F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4f.png"
    y_df = pd.read_csv("/home/phuong/data/phd-project/3--antigen/2--mouse-expt/y.csv")
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
