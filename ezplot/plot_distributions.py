from pathlib import Path

import fcsparser
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted

from ezplot.style import PALETTE, STYLE


def plot_fc_hist(fig_fp, cgry_df, gateline, class_labels, group_labels, xlabel, figsize=(32, 16), palette=PALETTE, rc_params=STYLE):
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
        cgry_df = cgry_df.loc[(cgry_df["response"] > 0)]
        fg = sns.FacetGrid(cgry_df, palette=palette, row="group", hue="repeat", aspect=7, height=2)
        fg.map_dataframe(sns.kdeplot, x="response", hue="class", fill=True, alpha=0.25, log_scale=True, palette=palette, linewidth=0)
        fg.map_dataframe(sns.kdeplot, x="response", hue="class", fill=False, alpha=1, log_scale=True, palette=palette, linewidth=1)
        fg.refline(y=0, linewidth=1, linestyle="-", color="#212121", clip_on=False)
        fg.refline(x=gateline, color="#212121", clip_on=False)
        handles = []
        for c in range(cgry_df["class"].nunique()):
            for g, ax in enumerate(fg.axes.ravel()):
                y_df = cgry_df.loc[(cgry_df["group"] == g) & (cgry_df["class"] == c)]
                tot_y = len(y_df)
                pos_y = np.sum(y_df["response"] > gateline)
                pct_y = np.around(100 * (pos_y / tot_y), 2)
                ax.text(1, c * 0.30 + 0.02, f"{pct_y}%", color=palette[c], fontsize=36, ha="right", transform=ax.transAxes)
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


def load_fcs(fcs_fp, channel=None):
    meta, data = fcsparser.parse(fcs_fp, meta_data_only=False, reformat_meta=True)
    if channel is not None:
        return data[channel].values
    return data


def analyze_cgry_fc_data(expt_dp, load_fnc, load_fnc_kwargs=None):
    df = []
    for c, class_dp in enumerate([dp for dp in natsorted(Path(expt_dp).glob("*")) if dp.is_dir()]):
        for g, group_dp in enumerate([dp for dp in natsorted(class_dp.glob("*")) if dp.is_dir()]):
            for r, rep_dp in enumerate([dp for dp in natsorted(group_dp.glob("*")) if dp.is_dir()]):
                fcs_fp = next(fp for fp in rep_dp.glob("*.fcs") if fp.is_file())
                y_col = load_fnc(fcs_fp, **load_fnc_kwargs)
                c_col = np.ones_like(y_col) * c
                g_col = np.ones_like(y_col) * g
                r_col = np.ones_like(y_col) * r
                i_df = pd.DataFrame({"class": c_col, "group": g_col, "repeat": r_col, "response": y_col})
                df.append(i_df)
    df = pd.concat(df)
    return df


def main():
    ## Figure 5B ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_5b.png"
    expt_dp = "/home/phuong/data/phd-project/4--antigen/0--K562-fc-staining/"
    cgry_df = analyze_cgry_fc_data(expt_dp, load_fnc=load_fcs, load_fnc_kwargs={"channel": "FL4_A"})
    group_labels = [
        "Plain K562",
        "Constitutive\nExpression",
        "Decoder + \nNone Input",
        "Decoder + \nSparse Input",
        "Decoder + \nDense Input",
    ]
    class_labels = [r"$\alpha$-CD19 AF647", r"$\alpha$-PSMA APC"]
    palette = ["#8069EC", "#EA822C"]
    xlabel = "Fluorescence (AU)"
    gateline = 3e3
    plot_fc_hist(fig_fp, cgry_df, gateline, class_labels, group_labels, xlabel, figsize=(32, 16), palette=palette)


if __name__ == "__main__":
    main()
