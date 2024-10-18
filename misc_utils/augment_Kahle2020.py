from itertools import combinations
from typing import Union, Tuple, Optional, Dict, Any, Literal

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pymatgen.core import Structure
from ase import units

from . import normal2lognormal as n2ln


# Groups are taken according to doi.org/10.1039/C9EE02457C (we split group B into two parts - B1 and B2,
# respectively, - according to the two tables in the supplementary information document)

# mapping from group to db id
GROUP_TO_IDS = dict(
    A={-1, 419852, 74950, 60850, 61338, 245988},
    B1={
        1008693, 421083, 1510745, 280992, 188009, 20032, 9000368, 428002, 2020217, 39761,
        245975, 1510933, 7035178, 193803, 33864, 8101456, 4329224, 2014117, 2208797,
    },
    B2={
        1530096, 61337, 73275, 60774, 2007413, 1004054, 83831, 34361, 1536985, 1535645, 23621, 2005920, 2019177,
        33953, 67234, 4321118, 60935, 75164, 1539516, 4329225, 425174,
    },
    C={
        34003, 4306193, 92468, 62137, 65260, 2012178, 245978, 291512, 174533, 1510140, 1511474, 1008009, 1501470,
        4317, 1510224, 4002768, 28526, 7024042, 61199, 2218562, 75071, 153620, 86184, 1535987, 1532734, 2220995,
        9014879, 642182, 34079, 2003027, 174443, 426103, 61218, 1511740, 7224138, 4337787, 73124, 1530934, 9007843,
        2000944, 1537475, 2242045, 50612, 1544389, 49022, 40247, 67236, 416888, 9007831, 246277, 7213712, 75031,
        69300, 4337786, 167518, 424281, 4330276, 87991, 4155, 405777, 7222190, 75516, 424352, 1528861, 2310701,
        7031897, 1100060, 1535227, 187751, 182033,
    },
    D={
        84602, 1534486, 1000333, 9004248, 423774, 416956, 1511223, 4326716,
        1530960, 1510187, 68251, 190656, 1526845, 1535801, 50232,
    }
)

# mapping from db id to group
ID_TO_GROUP = {id_i: k for k, v in GROUP_TO_IDS.items() for id_i in v}

def read_and_augment(
    input_csv_file: str, *,
    validation_plot_n2ln: Union[bool, str] = False,
    validation_plot_extrapolation: Union[bool, str] = False,
    max_KL: float = 0.3,
    id419852_policy: Literal["keep_both", "drop_small", "drop_large"] = "drop_large",
    drop_group_D: bool = True,
) -> pd.DataFrame:
    """
    Read Kahle2020 dataset in csv format, augment and return resulting pandas DataFrame

    Parameters
    ----------
    input_csv_file: str
        Path to input file
    validation_plot_n2nl: Union[bool, str]
        Whether to make validation plots for fitting lognorm parameters for diffusion coefficients
        (default = False). If str, interpreted as path to save the resulting figure.
    validation_plot_extrapolation: Union[bool, str]
        Whether to make validation plots for the D extrapolation procedure (default = False). If str,
        interpreted as path to save the resulting figure
    max_KL: float
        Maximal value of KL in the norm-lognorm fit to include structure for D extrapolation (default = 0.3)
    id419852_policy: Literal["keep_both", "drop_small", "drop_large"]
        Structure 419852 is present twice in the dataset (it's re-simulated with doubled super-cell). This
        parameter decides which one of them to keep. Defaults to "drop_large".
    drop_group_D: bool
        Whether to drop group D entries (the faulty ones). Defaults to true.

    Returns
    -------
    pd.DataFrame
        resulting augmented dataframe
    """
    df_Kahle2020 = pd.read_csv(input_csv_file)

    all_ids = set(df_Kahle2020["src_id"])

    ################
    #### CHECKS ####
    # 1) checking each group is entirely shared in the archive (this is not the case for group D)
    assert len(GROUP_TO_IDS["A"]) == len(GROUP_TO_IDS["A"].intersection(all_ids))
    assert len(GROUP_TO_IDS["B1"]) == len(GROUP_TO_IDS["B1"].intersection(all_ids))
    assert len(GROUP_TO_IDS["B2"]) == len(GROUP_TO_IDS["B2"].intersection(all_ids))
    assert len(GROUP_TO_IDS["C"]) == len(GROUP_TO_IDS["C"].intersection(all_ids))
    # For group D, 10 structures not reported in the archive:
    assert len(GROUP_TO_IDS["D"]) == len(GROUP_TO_IDS["D"].intersection(all_ids)) + 10

    # 2) checking each element from archive is in a group
    assert (
        GROUP_TO_IDS["A"]
        | GROUP_TO_IDS["B1"]
        | GROUP_TO_IDS["B2"]
        | GROUP_TO_IDS["C"]
        | GROUP_TO_IDS["D"].intersection(all_ids)
    ) == all_ids

    # 3) checking the groups don't intersect
    for (k1, v1), (k2, v2) in combinations(GROUP_TO_IDS.items(), 2):
        assert len(v1.intersection(v2)) == 0, (k1, k2)

    df_Kahle2020["group"] = df_Kahle2020["src_id"].apply(lambda x: ID_TO_GROUP[x])
    assert (
        df_Kahle2020["group"] == df_Kahle2020["src_id"].apply(
            lambda x: [k for k, v in GROUP_TO_IDS.items() if x in v][0]
        )
    ).all()

    D_mean = df_Kahle2020["diffusion_mean_cm2_s"]
    D_err = df_Kahle2020["diffusion_sem_cm2_s"]
    is_valid = ~D_err.isna()

    print("Estimating lognorm D parameters")
    (mu_lognorm, sigma_lognorm), vals_KL = n2ln.fit_lognorm2norm(
        D_mean.loc[is_valid], D_err.loc[is_valid], show_progress=True,
    )
    df_Kahle2020.loc[is_valid, "log_diffusion_mu"] = mu_lognorm
    df_Kahle2020.loc[is_valid, "log_diffusion_sigma"] = sigma_lognorm
    df_Kahle2020.loc[is_valid, "log_diffusion_KL"] = vals_KL

    if validation_plot_n2ln:
        import matplotlib.pyplot as plt

        num_plots = (df_Kahle2020["log_diffusion_KL"] <= max_KL).sum()
        ncols = int(np.ceil(np.sqrt(num_plots)))
        if ncols > 6:
            ncols = 6
        nrows = int(np.ceil(num_plots / ncols))
        fig = plt.figure(figsize=(20, 10 * nrows / ncols))

        iplot = 0
        for _, row in df_Kahle2020.sort_values("log_diffusion_KL", ascending=True).iterrows():
            if np.isnan(row["diffusion_sem_cm2_s"]): continue
            if row["log_diffusion_KL"] > max_KL: continue

            iplot += 1
            plt.subplot(nrows, ncols, iplot)
            n_mu = row["diffusion_mean_cm2_s"]
            n_sigma = row["diffusion_sem_cm2_s"]
            log10n_mu = row["log_diffusion_mu"] / np.log(10)
            log10n_sigma = row["log_diffusion_sigma"] / np.log(10)

            xx = np.linspace(
                n_mu - 5 * n_sigma,
                max(n_mu + 5 * n_sigma, n_sigma),
                1001,
            )
            xx2 = xx[xx > 0]
            plt.plot(xx, np.exp(_logprob_n(xx, n_mu, n_sigma)), label="orig. normal")
            plt.plot(xx2, np.exp(_logprob_log10n(xx2, log10n_mu, log10n_sigma)), label="fitted lognorm.")

            if n_mu > 0:
                log10n_prop_mu = np.log10(n_mu)
                log10n_prop_sigma = n_sigma / n_mu / np.log(10)
                plt.plot(xx2, np.exp(_logprob_log10n(xx2, log10n_prop_mu, log10n_prop_sigma)), label="propagated lognorm.")
            plt.yticks([])
            plt.text(
                xx[-1],
                plt.ylim()[1] * 0.95,
                f"{row['log_diffusion_KL']:.2}",
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
            )
            if iplot == 1:
                plt.legend(loc="upper left")
        plt.tight_layout()
        if isinstance(validation_plot_n2ln, str):
            fig.savefig(validation_plot_n2ln)

    extrap_data = []
    if validation_plot_extrapolation:
        num_plots = df_Kahle2020.groupby("first_frame_structure").apply(
            lambda g: ((g["log_diffusion_KL"].max() <= max_KL) and len(g) > 1)
        ).sum()
        ncols = int(np.ceil(np.sqrt(num_plots)))
        if ncols < 4: ncols = 4
        nrows = int(np.ceil(num_plots / ncols))

        import matplotlib.pyplot as plt
        fig_extrapolation = plt.figure(figsize=(15, 10 * nrows / ncols), dpi=100)
        ifig = 0

    for _, g in df_Kahle2020.sort_values("label").groupby("first_frame_structure", sort=False):
        if len(g) == 1: continue
        if g["log_diffusion_KL"].max() > max_KL: continue
        g = g.sort_values("temperature")
        assert len(g) == 4

        if validation_plot_extrapolation:
            ifig += 1
            plt.subplot(nrows, ncols, ifig)

        extrapolated_300K_logD_mu, extrapolated_300K_logD_sigma, chi2ndof = _process_structure(
            stru_data=g,
            max_KL=max_KL,
            visualization_params="default" if validation_plot_extrapolation else None,
        )
        def _get_unique(col: str):
            (val,) = g[col].unique()
            return val

        extrap_entry = {
            col: _get_unique(col) for col in [
                "starting_structure",
                "label",
                "src_database",
                "src_id",
                "first_frame_structure",
                "group",
            ]
        }
        extrap_entry["temperature"] = 300.0
        extrap_entry["log_diffusion_mu"] = extrapolated_300K_logD_mu
        extrap_entry["log_diffusion_sigma"] = extrapolated_300K_logD_sigma
        extrap_entry["log_diffusion_KL"] = g["log_diffusion_KL"].max()
        extrap_entry["extrapolation_chi2ndof"] = chi2ndof
        extrap_data.append(extrap_entry)

    if validation_plot_extrapolation:
        plt.tight_layout()
        if isinstance(validation_plot_extrapolation, str):
            fig_extrapolation.savefig(validation_plot_extrapolation)

    df_Kahle2020 = pd.concat(
        [
            df_Kahle2020,
            pd.DataFrame(extrap_data),
        ],
        axis=0,
    ).reset_index(drop=True)

    structs = df_Kahle2020["starting_structure"].apply(lambda x: Structure.from_str(x, "json"))
    df_Kahle2020["n_Li"] = structs.apply(lambda x: x.composition["Li"] / x.lattice.volume)
    D_to_sigma_factor = (
        df_Kahle2020["n_Li"]  # 1 / A^3
        / (df_Kahle2020["temperature"] * units.kB)  # eV
        * (1e24 / units.C)  # (A/cm)^3 * (e / C)
    )
    df_Kahle2020["NE_conductivity_S_cm"] = df_Kahle2020["diffusion_mean_cm2_s"] * D_to_sigma_factor  # S / cm
    df_Kahle2020["log_NE_conductivity_mu"] = df_Kahle2020["log_diffusion_mu"] + np.log(D_to_sigma_factor)

    df_Kahle2020["ffs_size"] = df_Kahle2020["first_frame_structure"].apply(
        lambda x: Structure.from_str(x, "json").num_sites
    )
    if id419852_policy != "keep_both":
        if id419852_policy == "drop_large":
            excl_size = 72
        elif id419852_policy == "drop_small":
            excl_size = 36
        else:
            raise NotImplementedError(f"Unexpected value for id419852_policy: {id419852_policy}")
        df_Kahle2020 = df_Kahle2020.query(f"(src_id != 419852) | (ffs_size != {excl_size})")

    if drop_group_D:
        df_Kahle2020 = df_Kahle2020.query("group != 'D'")

    return df_Kahle2020

def _logprob_log10n(x, m, s):
    return (
        -(np.log10(x) - m)**2 / 2 / s**2
        -np.log(x * s * np.log(10) * np.sqrt(2 * np.pi))
    )

def _logprob_n(x, mu, sigma):
    return (
        -(x - mu)**2 / 2 / sigma**2
        -np.log(sigma * np.sqrt(2 * np.pi))
    )

def _fit_slope_intercept(
    x: np.ndarray, y: np.ndarray, y_err: np.ndarray,
) -> Tuple[float, float, np.ndarray, float]:
    """
    Least squares fit (intended for fitting log10(D [cm^2/s]) vs 1/T [K^-1])

    Parameters
    ----------
    x: np.ndarray
        Inverse temperature values in K^-1
    y: np.ndarray
        log10(D [cm^2/s])
    y_err: np.ndarray
        error of y

    Returns
    -------
    slope: float
    intercept: float
    cov_matrix: np.ndarray
    chi2ndof: float
    """

    vars = [x, y, y_err]
    for i in range(len(vars)):
        if not isinstance(vars[i], np.ndarray):
            vars[i] = np.array(vars[i])

    assert vars[0].ndim == 1
    assert len(set(v.shape for v in vars)) == 1
    [x, y, y_err] = vars
    assert (y_err > 0).all()

    def _func(x, slope, intercept):
        return x * slope + intercept

    p_opt, p_cov, infodict, mesg, ier = curve_fit(
        f=_func, xdata=x, ydata=y, sigma=y_err, p0=(-1000, -2), full_output=True
    )

    return (*p_opt, p_cov, (infodict["fvec"]**2).sum() / (len(x) - 2))

_DEFAULT_VIS_PARAMS = dict(
    color="#1f77b4",
    xmin=0.8e-3,
    xmax=4.1e-3,
    ymin=-11.0,
    ymax=-3.0,
    xticks=[0.001, 0.002, 0.003, 0.004],
)

def _compare_composition(
    stru_small: str, stru_large: str,
) -> int:
    stru_small = Structure.from_str(stru_small, "json")
    stru_large = Structure.from_str(stru_large, "json")
    for i in range(1, 28):
        if stru_small.composition * i == stru_large.composition:
            return i
    assert False

def _process_structure(
    stru_data: pd.DataFrame, *,
    max_KL: float,
    visualization_params: Optional[Union[Dict[str, Any], Literal["default"]]] = None,
) -> Tuple[float, float, float]:
    """
    Extrapolate log D to lower temperatures.

    Returns
    -------
    extrapolated_300K_logD_mu: float
    extrapolated_300K_logD_sigma: float
    chi2ndof: float
    """

    assert len(stru_data) == 4
    stru_data = stru_data.sort_values("temperature")
    assert (stru_data["temperature"].to_numpy() == [500.0, 600.0, 750.0, 1000.0]).all()
    assert stru_data["log_diffusion_KL"].max() <= max_KL

    (label,) = stru_data["label"].unique()
    (ff_stru,) = stru_data["first_frame_structure"].unique()
    (s_stru,) = stru_data["starting_structure"].unique()
    supercell_size = _compare_composition(s_stru, ff_stru)

    x = (1.0 / stru_data["temperature"]).to_numpy()
    y = (stru_data["log_diffusion_mu"] / np.log(10)).to_numpy()
    y_err = (stru_data["log_diffusion_sigma"] / np.log(10)).to_numpy()
    (slope, intercept, cov, chi2ndof) = _fit_slope_intercept(
        x, y=y,
        y_err=y_err,
    )

    assert cov.shape == (2, 2)
    assert cov[0, 1] == cov[1, 0]
    assert np.linalg.det(cov) > 0

    inv_T = 1.0 / 300
    extrapolated_300K_log10D_mu = slope * inv_T + intercept
    extrapolated_300K_log10D_sigma = np.sqrt(
        inv_T**2 * cov[0, 0]
        + 2 * inv_T * cov[0, 1]
        + cov[1, 1]
    )

    if visualization_params is not None:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        if visualization_params == "default":
            visualization_params = _DEFAULT_VIS_PARAMS
        c = mpl.colors.to_hex(visualization_params["color"])
        plt.fill_between(
            x, y - y_err, y + y_err,
            facecolor=c + "66",
            edgecolor=c + "ff",
        )

        xx = np.linspace(visualization_params["xmin"], visualization_params["xmax"], 100)
        extrapolated_yy = slope * xx + intercept
        extrapolated_yy_err = np.sqrt(
            xx**2 * cov[0, 0]
            + 2 * xx * cov[0, 1]
            + cov[1, 1]
        )
        lower = extrapolated_yy - extrapolated_yy_err
        upper = extrapolated_yy + extrapolated_yy_err
        plt.plot(xx, lower, '--', color='k', alpha=0.6)
        plt.plot(xx, upper, '--', color='k', alpha=0.6)
        plt.text(
            0.98 * xx[-1] + 0.02 * xx[0],
            0.98 * visualization_params["ymax"] + 0.02 * visualization_params["ymin"],
            f"chi2/ndof: {chi2ndof:7.3}\nmax KL:    {stru_data['log_diffusion_KL'].max():7.3}",
            color="k",
            fontsize=7,
            horizontalalignment="right",
            verticalalignment="top",
            fontname="monospace",
        )

        plt.title(f"{label}" + ("" if supercell_size == 1 else f" ({supercell_size}-supercell)"))
        plt.xlim(visualization_params["xmin"], visualization_params["xmax"])
        plt.ylim(visualization_params["ymin"], visualization_params["ymax"])
        plt.xticks(visualization_params["xticks"])

    return (
        extrapolated_300K_log10D_mu * np.log(10),
        extrapolated_300K_log10D_sigma * np.log(10),
        chi2ndof,
    )