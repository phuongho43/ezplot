from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted

from ezplot.style import PALETTE, STYLE


def plot_crty(
    fig_fp, crty_df, class_labels, xlabel, ylabel, tu_df=None, figsize=(24, 16), lsizes=None, ldashes=None, leg_loc="best", leg_ncol=1, xticks=None, yticks=None, xlim=None, ylim=None, xlog=False, ylog=False, palette=PALETTE, rc_params=STYLE
):
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
                ax.axvspan(t, t + dt, color="#648FFF", lw=2)
            handles.append(mpl.lines.Line2D([], [], color="#648FFF", lw=lw, ls=(0, (1, 0))))
            class_labels.insert(0, "Input")
        lsizes = [lw] * crty_df["c"].nunique() if lsizes is None else lsizes
        ldashes = [(1, 0)] * crty_df["c"].nunique() if ldashes is None else ldashes
        sns.lineplot(ax=ax, data=crty_df, x="t", y="y", hue="c", size="c", sizes=lsizes, style="c", dashes=ldashes, errorbar=("se", 1.96), palette=palette)
        for i in range(crty_df["c"].nunique()):
            handle = mpl.lines.Line2D([], [], color=palette[i], lw=lw, ls=(0, ldashes[i]))
            handles.append(handle)
        ax.legend(handles, class_labels, loc=leg_loc, ncol=leg_ncol)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        if xticks is not None:
            ax.xaxis.set_ticks(xticks)
        if yticks is not None:
            ax.yaxis.set_ticks(yticks)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def plot_cxy(fig_fp, cxy_df, class_labels, xlabel="X", ylabel="Y", xticks=None, yticks=None, xlim=None, ylim=None, figsize=(24, 20), leg_loc="best", palette=PALETTE, rc_params=STYLE):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        handles = []
        sns.scatterplot(ax=ax, data=cxy_df, x="x", y="y", hue="c", s=800, linewidth=2, edgecolor="#212121", palette=palette, alpha=0.8, antialiased=True)
        for c in range(cxy_df["c"].nunique()):
            handle = mpl.lines.Line2D([], [], color=palette[c], marker="o", markersize=32, linewidth=0)
            handles.append(handle)
        ax.legend(handles, class_labels, loc=leg_loc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.xaxis.set_ticks(xticks)
        if yticks is not None:
            ax.yaxis.set_ticks(yticks)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def convert_u_ta_tb(u_ta_tb_df, t0, tf):
    u_ta = np.round(u_ta_tb_df["ta"].values, 1)
    u_tb = np.round(u_ta_tb_df["tb"].values, 1)
    tt = np.round(np.arange(t0, tf + 0.1, 0.1), 1)
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


def calc_dF_F0(y_df):
    F0 = y_df["y"].iloc[:5].mean()
    dF = y_df["y"] - F0
    y_df["y"] = dF / F0
    return y_df


def prep_rtz_data(class_dp, csv_fn="y.csv", calc_fnc=None, calc_fnc_kwargs=None):
    rt_ave = []
    rtz_df = []
    for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(Path(class_dp).glob("*")) if rep_dp.is_dir()]):
        csv_fp = rep_dp / csv_fn
        tz_df = pd.read_csv(csv_fp)
        calc_fnc_kwargs = {} if calc_fnc_kwargs is None else calc_fnc_kwargs
        tz_df = calc_fnc(tz_df, **calc_fnc_kwargs)
        tz_df["r"] = np.ones(len(tz_df), dtype=int) * r
        rtz_df.append(tz_df)
        rt_ave.append(tz_df["t"].values)
    for tz_df in rtz_df:
        tz_df["t"] = np.array(rt_ave).mean(axis=0)
    rtz_df = pd.concat(rtz_df)
    return rtz_df


def prep_crtz_data(class_dps):
    rty_dfs = []
    rtu_dfs = []
    for c, class_dp in enumerate([Path(dp) for dp in class_dps]):
        rty_df = prep_rtz_data(class_dp, csv_fn="results/y.csv", calc_fnc=calc_dF_F0)
        rty_df["c"] = np.ones(len(rty_df), dtype=int) * c
        rty_dfs.append(rty_df)
        t0 = rty_df["t"].iloc[0]
        tf = rty_df["t"].iloc[-1]
        rtu_df = prep_rtz_data(class_dp, csv_fn="u.csv", calc_fnc=convert_u_ta_tb, calc_fnc_kwargs={"t0": t0, "tf": tf})
        rtu_df["c"] = np.ones(len(rtu_df), dtype=int) * c
        rtu_dfs.append(rtu_df)
    crty_df = pd.concat(rty_dfs)
    crtu_df = pd.concat(rtu_dfs)
    return crty_df, crtu_df


def main():
    ## Figure 1D ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_1d.png"
    obj_addr_csv_fp = "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/obj_addr.csv"
    obj_addr_df = pd.read_csv(obj_addr_csv_fp)
    obj_gen1_df = obj_addr_df.loc[(obj_addr_df["gen_j"] == 0), ["obj_0", "obj_1"]]
    obj_gen1_df["c"] = np.ones(len(obj_gen1_df), dtype=int) * 0
    obj_gen10_df = obj_addr_df.loc[(obj_addr_df["gen_j"] == 9), ["obj_0", "obj_1"]]
    obj_gen10_df["c"] = np.ones(len(obj_gen10_df), dtype=int) * 1
    obj_gen100_df = obj_addr_df.loc[(obj_addr_df["gen_j"] == 99), ["obj_0", "obj_1"]]
    obj_gen100_df["c"] = np.ones(len(obj_gen100_df), dtype=int) * 2
    obj_pareto_df = obj_addr_df.loc[(obj_addr_df["is_pareto"]), ["obj_0", "obj_1"]]
    obj_pareto_df["c"] = np.ones(len(obj_pareto_df), dtype=int) * 3
    cxy_df = pd.concat([obj_gen1_df, obj_gen10_df, obj_gen100_df, obj_pareto_df])
    cxy_df.columns = ["x", "y", "c"]
    class_labels = ["Gen 1", "Gen 10", "Gen 100", "Best (Pareto)"]
    xlabel = "Simplicity"
    ylabel = "Performance"
    xticks = np.arange(0.0, 1.1, 0.2)
    yticks = np.arange(0.0, 1.1, 0.2)
    xlim = (-0.1, 1.1)
    ylim = (-0.1, 1.1)
    palette = ["#2ECC71", "#F1C40F", "#EA822C", "#D143A4"]
    plot_cxy(fig_fp, cxy_df, class_labels, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, palette=palette)

    ## Figure 1E ##
    fig_fps = ["/home/phuong/data/phd-project/figures/fig_1e_0.png", "/home/phuong/data/phd-project/figures/fig_1e_1.png", "/home/phuong/data/phd-project/figures/fig_1e_2.png"]
    crty_csv_fps = [
        "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_motif_high_low.csv",
        "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_motif_low_high.csv",
        "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_motif_high_high.csv",
    ]
    for fig_fp, crty_csv_fp in zip(fig_fps, crty_csv_fps):
        crty_df = pd.read_csv(crty_csv_fp)
        tu = np.arange(0, 121, 1.0)
        uu = np.zeros_like(tu)
        uu[40:80:10] = 1.0
        uu[80:121:1] = 1.0
        tu_df = pd.DataFrame({"t": tu, "u": uu})
        class_labels = ["Dense\nDecoder", "Sparse\nDecoder"]
        xlabel = "Time (AU)"
        ylabel = "Output (AU)"
        lsizes = [16, 10]
        palette = ["#8069EC", "#EA822C"]
        plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, figsize=(24, 20), lsizes=lsizes, leg_loc="best", palette=palette)

    ## Figure 1F ##
    fig_fps = ["/home/phuong/data/phd-project/figures/fig_1f_0.png", "/home/phuong/data/phd-project/figures/fig_1f_1.png", "/home/phuong/data/phd-project/figures/fig_1f_2.png"]
    crty_csv_fps = [
        "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_motif_no_x2_self_activ.csv",
        "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_motif_high_x2_induc.csv",
        "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_motif_no_x2_self_activ_high_x2_induc.csv",
    ]
    ref_crty_csv_fp = "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_motif_high_high.csv"
    ref_crty_df = pd.read_csv(ref_crty_csv_fp)
    ref_crty_df = ref_crty_df.loc[(ref_crty_df["c"] == 1)]
    ref_crty_df["c"] = np.ones(len(ref_crty_df), dtype=int) * 0
    for fig_fp, test_crty_csv_fp in zip(fig_fps, crty_csv_fps):
        test_crty_df = pd.read_csv(test_crty_csv_fp)
        test_crty_df = test_crty_df.loc[(test_crty_df["c"] == 1)]
        crty_df = pd.concat([ref_crty_df, test_crty_df])
        tu = np.arange(0, 121, 1.0)
        uu = np.zeros_like(tu)
        uu[40:80:10] = 1.0
        uu[80:121:1] = 1.0
        tu_df = pd.DataFrame({"t": tu, "u": uu})
        class_labels = ["Regular", "Modified"]
        xlabel = "Time (AU)"
        ylabel = "Output (AU)"
        lsizes = [16, 10]
        palette = ["#EA822C", "#34495E"]
        plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, figsize=(24, 20), lsizes=lsizes, leg_loc="best", palette=palette)

    ## Figure 1G ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_1g.png"
    crty_csv_fp = "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_ave_response.csv"
    crty_df = pd.read_csv(crty_csv_fp)
    class_labels = ["Dense Decoder", "Sparse Decoder"]
    xlabel = "Input Pulse Freq. (Hz)"
    ylabel = "Mean Output (AU)"
    ylim = [-0.1, 1.1]
    palette = ["#8069EC", "#EA822C"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, figsize=(24, 20), lsizes=lsizes, ylim=ylim, xlog=True, leg_loc="upper left", palette=palette)

    ## Figure 2C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2c.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/",
    ]
    crty_df, crtu_df = prep_crtz_data(class_dps)
    tu_df = crtu_df.groupby("t", as_index=False)["u"].mean()
    class_labels = [
        "ddFP",
        "LOVfast",
        "LOVslow",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    lsizes = [16, 12, 8]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, lsizes=lsizes, palette=palette)

    ## Figure S1A ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1a.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/1--LOVfast-BL200uW/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/0--LOVfast-BL20uW/",
    ]
    crty_df, crtu_df = prep_crtz_data(class_dps)
    crty_df = crty_df.loc[(crty_df["t"] < 181)]
    tu_df = crtu_df.groupby("t", as_index=False)["u"].mean()
    tu_df = tu_df.loc[(tu_df["t"] < 181)]
    class_labels = [
        r"$\mathdefault{200\ \mu W/mm^2}$",
        r"$\mathdefault{20\ \mu W/mm^2}$",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#EA822C", "#2ECC71"]
    lsizes = [16, 12]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, lsizes=lsizes, palette=palette)

    ## Figure 2F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2f.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/1--V416I/",
    ]
    crty_df, crtu_df = prep_crtz_data(class_dps)
    tu_df = crtu_df.groupby("t", as_index=False)["u"].mean()
    class_labels = [
        "ddFP",
        "iLIDfast",
        "iLIDslow",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#34495E", "#2ECC71", "#D143A4"]
    lsizes = [16, 12, 8]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, lsizes=lsizes, palette=palette)

    ## Figure S1C ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_s1c.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/4--linker/0--iLIDslow-13AA/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/1--iLIDslow-20AA/",
    ]
    crty_df, crtu_df = prep_crtz_data(class_dps)
    crty_df = crty_df.loc[(crty_df["t"] < 181)]
    tu_df = crtu_df.groupby("t", as_index=False)["u"].mean()
    tu_df = tu_df.loc[(tu_df["t"] < 181)]
    class_labels = [
        "13 AA Linker",
        "20 AA Linker",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#2ECC71", "#EA822C"]
    lsizes = [16, 12]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, lsizes=lsizes, palette=palette)

    # Figure 2I ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2i.png"
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/",
    ]
    crty_df, crtu_df = prep_crtz_data(class_dps)
    crty_df = crty_df.loc[(crty_df["t"] < 155)]
    tu_df = crtu_df.groupby("t", as_index=False)["u"].mean()
    tu_df = tu_df.loc[(tu_df["t"] < 155)]
    class_labels = [
        "Sparse-ddFP",
    ]
    xlabel = "Time (s)"
    ylabel = r"$\mathbf{\Delta F/F_{0}}$"
    palette = ["#EA822C"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, leg_loc="lower left", palette=palette)

    ## Figure 3H ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_3h.png"
    psn_crty_csv_fp = "/home/phuong/data/phd-project/0--protosignet/0--dual-fm/results/fm_ave_response.csv"
    opi_crty_csv_fp = "/home/phuong/data/phd-project/2--optopi/4--fm-ave-response/fm_ave_response.csv"
    psn_crty_df = pd.read_csv(psn_crty_csv_fp)
    opi_crty_df = pd.read_csv(opi_crty_csv_fp)
    opi_crty_df["c"] += 2
    crty_df = pd.concat([psn_crty_df, opi_crty_df])
    rescale_01 = lambda x: (x - x.min()) / (x.max() - x.min())
    for c in range(crty_df["c"].nunique()):
        crty_df.loc[(crty_df["c"] == c), "y"] = rescale_01(crty_df.loc[(crty_df["c"] == c), "y"])
    class_labels = ["Dense Decoder (PSN)", "Sparse Decoder (PSN)", "Dense Decoder (OPI)", "Sparse Decoder (OPI)"]
    xlabel = "Input Pulse Freq. (Hz)"
    ylabel = "Norm. Output (AU)"
    ldashes = [(1, 1), (1, 1), (1, 0), (1, 0)]
    ylim = [-0.1, 1.5]
    yticks = np.arange(0.0, 1.1, 0.2)
    palette = ["#8069EC", "#EA822C", "#8069EC", "#EA822C"]
    plot_crty(fig_fp, crty_df, class_labels, xlabel=xlabel, ylabel=ylabel, figsize=(30, 20), ldashes=ldashes, yticks=yticks, ylim=ylim, xlog=True, leg_loc="best", leg_ncol=2, palette=palette)

    ## Figure 5F ##
    fig_fp = "/home/phuong/data/phd-project/figures/fig_5f.png"
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
    palette = ["#8069EC", "#EA822C", "#8069EC", "#EA822C"]
    ldashes = [(1, 0), (1, 1), (1, 1), (1, 0)]
    plot_crty(fig_fp, cty_df, class_labels, xlabel=xlabel, ylabel=ylabel, tu_df=tu_df, ldashes=ldashes, palette=palette)

    print("\a")


if __name__ == "__main__":
    main()
