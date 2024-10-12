import pandas as pd
import numpy as np


def join_data_and_preds_Kahle2020(
    df_preds: pd.DataFrame,
    df_data: pd.DataFrame,
) -> pd.DataFrame:
    df_preds = df_preds.copy()
    df_data = df_data.copy()

    df_data_ids = df_data.query("temperature == 1000.0")["src_id"]

    # validation + excluding predictions for group D structures
    assert len(df_preds) == len(df_preds["src_id"].unique())
    assert len(df_data_ids) == len(df_preds) - 5

    df_preds = df_preds.set_index("src_id").loc[df_data_ids].reset_index()

    assert len(df_preds) == len(df_preds["src_id"].unique())
    assert len(df_data_ids) == len(df_preds)

    assert set(df_data.groupby("src_id").apply(len).unique()) == {1, 4, 5}

    def _get_unique(x):
        unique_vals = np.unique(x)
        if not len(unique_vals):
            return float("NaN")
        (val,) = unique_vals
        return val

    df_targets = df_data.groupby("src_id").apply(
        lambda g: pd.Series(dict(
            D1000=_get_unique(g.query("temperature == 1000.0")["diffusion_mean_cm2_s"]),
            D1000_err=_get_unique(g.query("temperature == 1000.0")["diffusion_sem_cm2_s"]),
            log10D300=_get_unique(
                g.query("(temperature == 300.0) & (extrapolation_chi2ndof <= 3.5)")["log_diffusion_mu"] / np.log(10)
            ),
            log10D300_err=_get_unique(
                g.query("(temperature == 300.0) & (extrapolation_chi2ndof <= 3.5)")["log_diffusion_sigma"] / np.log(10)
            ),
            condNE1000=_get_unique(g.query("temperature == 1000.0")["NE_conductivity_S_cm"]),
            log10condNE300=_get_unique(
                g.query("(temperature == 300.0) & (extrapolation_chi2ndof <= 3.5)")["log_NE_conductivity_mu"] / np.log(10)
            ),
            group=_get_unique(g["group"]),
        ))
    ).reset_index()
    df_targets["log10condNE300_err"] = df_targets["log10D300_err"]
    df_targets["condNE1000_err"] = (df_targets["condNE1000"] * df_targets["D1000_err"] / df_targets["D1000"]).abs()

    df_preds = df_preds.join(df_targets.set_index("src_id"), on="src_id")
    return df_preds

def join_data_and_preds_exp(
    df_preds_full_mp: pd.DataFrame,
    df_data_exp_mp: pd.DataFrame,
) -> pd.DataFrame:
    df_preds_mp_exp = df_preds_full_mp.set_index("material_id").loc[df_data_exp_mp["mp"]].reset_index().copy()
    df_preds_mp_exp[["sigma_S_cm", "Ea"]] = df_data_exp_mp[["σ(RT)(S cm-1)", "Ea (eV)"]]
    return df_preds_mp_exp
