from pathlib import Path

import fcsparser
import numpy as np
import pandas as pd
from natsort import natsorted


def convert_u_ta_tb(tt, u_csv_fp):
    u_df = pd.read_csv(u_csv_fp)
    u_ta = np.round(u_df["ta"].values, 1)
    u_tb = np.round(u_df["tb"].values, 1)
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
    u_df = pd.DataFrame({"t": tt, "u": uu})
    return u_df


def load_fcs(fcs_fp, channel=None):
    meta, data = fcsparser.parse(fcs_fp, meta_data_only=False, reformat_meta=True)
    if channel is not None:
        return data[channel].values
    return data


def combine_rcg(data_dp, load_fnc, load_fnc_kwargs=None, save_csv_fp=None):
    dfs = []
    for r, repeat_dp in enumerate(natsorted(Path(data_dp).glob("*"))):
        for c, class_dp in enumerate(natsorted(Path(repeat_dp).glob("*"))):
            for g, group_fp in enumerate(natsorted(Path(class_dp).glob("*"))):
                y_col = load_fnc(group_fp, **load_fnc_kwargs)
                r_col = np.ones_like(y_col) * r
                c_col = np.ones_like(y_col) * c
                g_col = np.ones_like(y_col) * g
                i_df = pd.DataFrame({"repeat": r_col, "class": c_col, "group": g_col, "response": y_col})
                dfs.append(i_df)
    dfs = pd.concat(dfs)
    if save_csv_fp is not None:
        dfs.to_csv(save_csv_fp, index=False)
    return dfs


def calc_log2_norm(rcgy_df):
    for c in rcgy_df["class"].unique():
        for r in rcgy_df["repeat"].unique():
            ygi = rcgy_df.loc[(rcgy_df["class"] == c) & (rcgy_df["repeat"] == r), "response"]
            rcgy_df.loc[(rcgy_df["class"] == c) & (rcgy_df["repeat"] == r), "response"] = np.log2(ygi / ygi.iloc[0])
    return rcgy_df
