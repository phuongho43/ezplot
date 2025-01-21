from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ezplot.style import PALETTE, STYLE


def plot_fc_hist(fig_fp, y_df, gateline, class_labels, group_labels, xlabel, figsize=(32, 16), palette=PALETTE, rc_params=STYLE):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        sns.set_theme(
            style="ticks",
            rc={
                "axes.facecolor": (0, 0, 0, 0),
                "figure.figsize": figsize,
                "xtick.major.size": 12,
                "xtick.minor.size": 8,
                "xtick.major.width": 3,
                "xtick.minor.width": 2,
                "xtick.labelsize": 36,
                "legend.facecolor": "white",
                "legend.fontsize": 36,
            },
        )
        y_df = y_df.loc[(y_df["response"] > 0)]
        fg = sns.FacetGrid(y_df, palette=palette, row="group", hue="repeat", aspect=7, height=2)
        fg.map_dataframe(sns.kdeplot, x="response", hue="class", fill=True, alpha=0.25, log_scale=True, palette=palette, linewidth=0)
        fg.map_dataframe(sns.kdeplot, x="response", hue="class", fill=False, alpha=1, log_scale=True, palette=palette, linewidth=1)
        fg.refline(y=0, linewidth=1, linestyle="-", color="#212121", clip_on=False)
        fg.refline(x=gateline, color="#212121", clip_on=False)
        handles = []
        for c in range(y_df["class"].nunique()):
            for g, ax in enumerate(fg.axes.ravel()):
                ycg_df = y_df.loc[(y_df["group"] == g) & (y_df["class"] == c)]
                tot_ycg = len(ycg_df)
                pos_ycg = np.sum(ycg_df["response"] > gateline)
                pct_ycg = np.around(100 * (pos_ycg / tot_ycg), 2)
                ax.text(1, c * 0.30 + 0.02, f"{pct_ycg}%", color=palette[c], fontsize=36, ha="right", transform=ax.transAxes)
                ax.text(0, 0.02, f"{group_labels[g]}", color="#212121", fontsize=36, ha="left", transform=ax.transAxes)
            handles.append(mpl.lines.Line2D([], [], color=palette[c], lw=6))
        fg.axes.ravel()[0].legend(handles, class_labels, loc=(0, 0.62))
        fg.fig.subplots_adjust(hspace=0)
        fg.set_titles("")
        fg.set(yticks=[], ylabel="", xlim=(1, None))
        fg.set_xlabels(xlabel, fontsize=48, fontweight="bold")
        fg.despine(left=True, bottom=True)
        fg.fig.savefig(fig_fp)
        plt.close("all")


def main():
    # save_csv_fp = "/home/phuong/data/phd-project/3--antigen/0--K562-fc-staining/y.csv"
    # data_dp = "/home/phuong/data/phd-project/3--antigen/0--K562-fc-staining/data/"
    # combine_rcg(data_dp, load_fnc=load_fcs, load_fnc_kwargs={"channel": "FL4_A"}, save_csv_fp=save_csv_fp)

    fig_fp = "/home/phuong/data/phd-project/figures/fig_4b.png"
    y_csv_fp = "/home/phuong/data/phd-project/3--antigen/0--K562-fc-staining/y.csv"
    y_df = pd.read_csv(y_csv_fp)
    group_labels = [
        "Plain K562",
        "Constitutive",
        "Decoder + \nNone Input",
        "Decoder + \nSparse Input",
        "Decoder + \nDense Input",
    ]
    class_labels = [r"$\alpha$-CD19 AF647", r"$\alpha$-PSMA APC"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = "Fluorescence (AU)"
    gateline = 3e3
    plot_fc_hist(fig_fp, y_df, gateline, class_labels, group_labels, xlabel, figsize=(32, 16), palette=palette)


if __name__ == "__main__":
    main()
