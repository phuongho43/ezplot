from pathlib import Path

import fcsparser
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fluoressential.style import PALETTE, STYLE_LARGE
from natsort import natsorted


def load_fc_data(fcs_fp):
    meta, data = fcsparser.parse(fcs_fp, meta_data_only=False, reformat_meta=True)
    return data


def process_fc_data(save_fp, data_dp):
    y_df = pd.DataFrame()
    for r, repeat_dp in enumerate(natsorted(Path(data_dp).glob("*"))):
        for c, class_dp in enumerate(natsorted(Path(repeat_dp).glob("*"))):
            for g, fcs_fp in enumerate(natsorted(Path(class_dp).glob("*"))):
                fc_data = load_fc_data(fcs_fp)
                y_col = fc_data["FL4_A"].values
                r_col = np.ones_like(y_col) * r
                c_col = np.ones_like(y_col) * c
                g_col = np.ones_like(y_col) * g
                y_df_i = pd.DataFrame({"repeat": r_col, "class": c_col, "group": g_col, "response": y_col})
                y_df = pd.concat([y_df, y_df_i])
    y_df.to_csv(Path(save_fp), index=False)
    return y_df


def plot_fc_hist(fig_fp, y_csv_fp, gateline, class_labels, group_labels, xlabel, figsize=(32, 16), palette=PALETTE, rc_params=STYLE_LARGE):
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
        y_df = pd.read_csv(Path(y_csv_fp))
        y_df = y_df.loc[(y_df["response"] > 0)]
        fg = sns.FacetGrid(y_df, palette=palette, row="group", hue="repeat", aspect=7, height=2)
        fg.map_dataframe(sns.kdeplot, x="response", hue="class", fill=True, alpha=0.25, log_scale=True, palette=palette, linewidth=0)
        fg.map_dataframe(sns.kdeplot, x="response", hue="class", fill=False, alpha=1, log_scale=True, palette=palette, linewidth=1)
        fg.refline(y=0, linewidth=1, linestyle="-", color="#212121", clip_on=False)
        fg.refline(x=gateline, color="#212121", clip_on=False)
        for g, ax in enumerate(fg.axes.ravel()):
            yg_c0_df = y_df.loc[(y_df["group"] == g) & (y_df["class"] == 0)]
            yg_c1_df = y_df.loc[(y_df["group"] == g) & (y_df["class"] == 1)]
            tot_c0 = len(yg_c0_df)
            tot_c1 = len(yg_c1_df)
            pos_c0 = np.sum(yg_c0_df["response"] > gateline)
            pos_c1 = np.sum(yg_c1_df["response"] > gateline)
            pct_c0 = np.around(100 * (pos_c0 / tot_c0), 2)
            pct_c1 = np.around(100 * (pos_c1 / tot_c1), 2)
            ax.text(1, 0.30, f"{pct_c0}%", color=palette[0], fontsize=36, ha="right", transform=ax.transAxes)
            ax.text(1, 0.02, f"{pct_c1}%", color=palette[1], fontsize=36, ha="right", transform=ax.transAxes)
            ax.text(0, 0.02, f"{group_labels[g]}", color="#212121", fontsize=36, ha="left", transform=ax.transAxes)
            if g >= 2:
                ax.text(0, 0.30, "Decoder +", color="#212121", fontsize=36, ha="left", transform=ax.transAxes)
        handles = [
            mpl.lines.Line2D([], [], color=palette[0], lw=6),
            mpl.lines.Line2D([], [], color=palette[1], lw=6),
        ]
        fg.axes.ravel()[0].legend(handles, class_labels, loc=(0, 0.4))
        fg.fig.subplots_adjust(hspace=0)
        fg.set_titles("")
        fg.set(yticks=[], ylabel="", xlim=(1, 1e7))
        fg.set_xlabels(xlabel, fontsize=48, fontweight="bold")
        fg.despine(left=True, bottom=True)
        fg.fig.savefig(fig_fp)
        plt.close("all")


def main():
    # save_fp = "/home/phuong/data/phd-project/3--antigen/0--K562-fc-staining/y.csv"
    # data_dp = "/home/phuong/data/phd-project/3--antigen/0--K562-fc-staining/data/"
    # process_fc_data(save_fp, data_dp)

    fig_fp = "/home/phuong/data/phd-project/3--antigen/0--K562-fc-staining/y.png"
    y_csv_fp = "/home/phuong/data/phd-project/3--antigen/0--K562-fc-staining/y.csv"
    group_labels = [r"$\mathdefault{Plain\ K562}$", r"$\mathdefault{Constitutive}$", r"$\mathdefault{None\ Input}$", r"$\mathdefault{Sparse\ Input}$", r"$\mathdefault{Dense\ Input}$"]
    class_labels = [r"$\alpha$-CD19 AF647", r"$\alpha$-PSMA APC"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = "Fluorescence (AU)"
    fig_fp = "/home/phuong/data/phd-project/figures/fig_4b.png"
    gateline = 3e3
    plot_fc_hist(fig_fp, y_csv_fp, gateline, class_labels, group_labels, xlabel, figsize=(32, 16), palette=palette, rc_params=STYLE_LARGE)


if __name__ == "__main__":
    main()
