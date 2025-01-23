from pathlib import Path

import fcsparser
import numpy as np
import pandas as pd
from natsort import natsorted


def combine_csv(i_csv_fps, new_col, save_csv_fp=None):
    dfs = []
    for i, i_csv_fp in enumerate(i_csv_fps):
        i_df = pd.read_csv(i_csv_fp)
        i_df[new_col] = np.ones(len(i_df), dtype=int) * i
        dfs.append(i_df)
    dfs = pd.concat(dfs)
    if save_csv_fp is not None:
        dfs.to_csv(save_csv_fp, index=False)
    return dfs


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


def calc_tspan_aves(rty_df, tspans):
    aves_df = []
    for g, [t1, t2] in enumerate(tspans):
        ave_df = rty_df.loc[(rty_df["t"] >= t1) & (rty_df["t"] < t2), ["r", "y"]].copy()
        ave_df = ave_df.groupby("r", as_index=False)["y"].mean()
        ave_df["g"] = np.ones_like(ave_df["y"]) * g
        aves_df.append(ave_df)
    aves_df = pd.concat(aves_df)
    aves_df.rename(columns={"r": "repeat", "y": "response", "g": "group"}, inplace=True)
    return aves_df


def calc_log2_norm(rcgy_df):
    for c in rcgy_df["class"].unique():
        for r in rcgy_df["repeat"].unique():
            ygi = rcgy_df.loc[(rcgy_df["class"] == c) & (rcgy_df["repeat"] == r), "response"]
            rcgy_df.loc[(rcgy_df["class"] == c) & (rcgy_df["repeat"] == r), "response"] = np.log2(ygi / ygi.iloc[0])
    return rcgy_df
