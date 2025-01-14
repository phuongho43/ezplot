from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fluoressential.style import PALETTE, STYLE, STYLE_LARGE


def plot_dynamics(fig_fp, y_csv_fps, group_labels, u_csv_fp=None, xlabel="Time", ylabel="AU", figsize=(24, 16), palette=PALETTE, rc_params=STYLE):
    """Plot dynamics timeseries data.

    Args:
        fig_fp (str): absolute path for saving generated figure
        y_csv_fps (list[string]): absolute paths of CSV files with columns [t, y] describing output response over time
        u_csv_fp (str): absolute path of CSV file with columns [t, u] describing stimuli input over time
    """
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        handles = []
        lw = rc_params["lines.linewidth"]
        if u_csv_fp is not None:
            u_df = pd.read_csv(u_csv_fp)
            u_df = u_df.groupby("t", as_index=False)["u"].agg(["mean"])
            tu = u_df["t"].to_numpy()
            uu = u_df["mean"].to_numpy()
            for t in tu[uu > 0]:
                dt = np.mean(np.diff(tu))
                ax.axvspan(t, t + dt, color="#648FFF", alpha=0.8, lw=0)
            handles.append(mpl.lines.Line2D([], [], color="#648FFF", lw=lw, alpha=0.8, solid_capstyle="projecting"))
            group_labels.insert(0, "Input")
        for i, y_csv_fp in enumerate(y_csv_fps):
            y_df = pd.read_csv(y_csv_fp)
            sns.lineplot(ax=ax, data=y_df, x="t", y="y", errorbar=("se", 1.96), color=palette[i], lw=lw)
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
    # fig_fp = "/home/phuong/data/phd-project/figures/fig_2c.png"
    # y_csv_fps = [
    #     "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/y.csv",
    #     "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/results/y.csv",
    #     "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/results/y.csv",
    # ]
    # group_labels = [
    #     "ddFP",
    #     "LOVfast",
    #     "LOVslow",
    # ]
    # u_csv_fp = "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/u.csv"
    # xlabel = "Time (s)"
    # ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    # palette = ["#34495E", "#2ECC71", "#D143A4"]
    # plot_dynamics(fig_fp, y_csv_fps, group_labels, u_csv_fp, xlabel=xlabel, ylabel=ylabel, palette=palette, rc_params=STYLE_LARGE)

    # fig_fp = "/home/phuong/data/phd-project/figures/fig_2f.png"
    # y_csv_fps = [
    #     "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/y.csv",
    #     "/home/phuong/data/phd-project/1--biosensor/2--LID/0--I427V/results/y.csv",
    #     "/home/phuong/data/phd-project/1--biosensor/2--LID/1--V416I/results/y.csv",
    # ]
    # group_labels = [
    #     "ddFP",
    #     "LIDfast",
    #     "LIDslow",
    # ]
    # u_csv_fp = "/home/phuong/data/phd-project/1--biosensor/0--ddFP/results/u.csv"
    # xlabel = "Time (s)"
    # ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    # palette = ["#34495E", "#2ECC71", "#D143A4"]
    # plot_dynamics(fig_fp, y_csv_fps, group_labels, u_csv_fp, xlabel=xlabel, ylabel=ylabel, palette=palette, rc_params=STYLE_LARGE)

    # fig_fp = "/home/phuong/data/phd-project/figures/fig_2i.png"
    # y_csv_fps = [
    #     "/home/phuong/data/phd-project/1--biosensor/3--sparser/results/y.csv",
    # ]
    # u_csv_fp = "/home/phuong/data/phd-project/1--biosensor/3--sparser/results/u.csv"
    # group_labels = [
    #     "Sparse\nDecoder",
    # ]
    # xlabel = "Time (s)"
    # ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    # palette = ["#EA822C"]
    # plot_dynamics(fig_fp, y_csv_fps, group_labels, u_csv_fp, xlabel=xlabel, ylabel=ylabel, palette=palette, rc_params=STYLE_LARGE)

    fig_fp = "/home/phuong/data/phd-project/figures/fig_4f.png"
    y_csv_fps = [
        "/home/phuong/data/phd-project/3--antigen/2--mouse-expt/0.csv",
        "/home/phuong/data/phd-project/3--antigen/2--mouse-expt/1.csv",
        "/home/phuong/data/phd-project/3--antigen/2--mouse-expt/2.csv",
        "/home/phuong/data/phd-project/3--antigen/2--mouse-expt/3.csv",
    ]
    u_csv_fp = None
    group_labels = [
        "Dense-CD19 (Dense Input)",
        "Sparse-PSMA (Dense Input)",
        "Dense-CD19 (Sparse Input)",
        "Sparse-PSMA (Sparse Input)",
    ]
    xlabel = "Time (d)"
    ylabel = "Total Flux (p/s)"
    palette = ["#8069EC", "#EA822C", "#8069EC", "#EA822C"]
    plot_dynamics(fig_fp, y_csv_fps, group_labels, u_csv_fp, xlabel=xlabel, ylabel=ylabel, palette=palette, rc_params=STYLE_LARGE)


if __name__ == "__main__":
    main()
