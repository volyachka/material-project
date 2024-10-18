from typing import Iterable, List, Tuple, Literal, Optional, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm


def get_features_meta_info(
    masking_options: Iterable[str] = ("", "masked1p5_", "Wmasked1p5_"),
    connection_options: Iterable[str] = ("connected", "disconnected"),
    fv_thresholds: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    barrier_percentiles: Iterable[float] = (0.00, 0.03, 0.05, 0.10),
) -> pd.DataFrame:
    barrier_features_meta_info = pd.DataFrame([
            dict(
                feature="barrier" + ("" if percentile == 0.0 else f"_robust_{percentile:04.2f}".replace(".", "p")),
                level=percentile,
                type="barrier",
                weighted_direction=-1.0,
            ) for percentile in barrier_percentiles
    ]).set_index("feature")
    fv_features_meta_info = pd.DataFrame([
            dict(
                feature=f"{masked}fv_{thr:03.1f}_{connectedness}_union".replace(".", "p"),
                level=thr,
                type=f"{masked}fv_{connectedness}",
                weighted_direction=1.0,
            )
            for masked in masking_options
            for connectedness in connection_options
            for thr in fv_thresholds
            if not (  # exclude really bad features
                masked in ["masked1p5_", "Wmasked1p5_"]
                and connectedness == "connected"
                and (thr < 0.5)
            )
    ]).set_index("feature")

    return pd.concat([barrier_features_meta_info, fv_features_meta_info], axis=0)

def add_feature(
    dfs: List[pd.DataFrame],
    features_meta_info: pd.DataFrame,
    func: Callable,
    name: str,
    type: str,
    level: float,
    weighted_direction: float = 1.0,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    assert name not in features_meta_info.index
    dfs = [df.copy() for df in dfs]
    for df in dfs:
        assert name not in df.columns
        df[name] = func(df)
    features_meta_info = pd.concat([
        features_meta_info,
        pd.DataFrame([
            dict(
                feature=name,
                level=level,
                type=type,
                weighted_direction=weighted_direction,
            )
        ]).set_index("feature"),
    ], axis=0)
    return dfs, features_meta_info


def add_feature_np(
    dfs: List[pd.DataFrame],
    features_meta_info: pd.DataFrame,
    values: List[np.array],
    name: str,
    type: str,
    level: float,
    weighted_direction: float = 1.0,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    assert name not in features_meta_info.index
    dfs = [df.copy() for df in dfs]
    for df, value in zip(dfs, values):
        assert name not in df.columns
        df[name] = value
    features_meta_info = pd.concat([
        features_meta_info,
        pd.DataFrame([
            dict(
                feature=name,
                level=level,
                type=type,
                weighted_direction=weighted_direction,
            )
        ]).set_index("feature"),
    ], axis=0)
    return dfs, features_meta_info


def plot_features_Kahle2020(
    features: List[str],
    preds_df: pd.DataFrame,
    transform_barriers: bool = True,
    base_target_column: str = "condNE1000",
    base_target_thr: float = 1e-2,
    base_target_clip_low: float = 1e-7,
    base_target_lims: Tuple[float] = (-7.5, 4.5),
    extrap_target_column: str = "log10condNE300",
    inset_ylabel: str = "extrap",
    main_ylabel: str = "log10(cond.NE @ 1000K [S / cm])",
    ncols: int = 3,
    fig: Optional[plt.Figure] = None,
    xlabel_map: Optional[dict] = None,
    main_fill_colored: bool = False,
    triangle_clipped: bool = False,
) -> plt.Figure:
    preds_df = preds_df.copy()
    nrows = int(np.ceil(len(features) / ncols))
    if fig is None:
        fig = plt.figure(figsize=(16, 4 * nrows))
    else:
        fig = plt.figure(fig)

    for iplot, col in enumerate(features, 1):

        if transform_barriers:
            _f0 = (lambda x: 1.0 / x) if "barrier" in col else (lambda x: x)
            _f = lambda x: np.clip(
                _f0(x),
                a_min=_f0(preds_df[col]).loc[_f0(preds_df[col]) > 0].min() / 10,
                a_max=None,
            )
        else:
            _f = lambda x: x

        xlims = _f(preds_df[col])
        xlims = (xlims.min() / 2, xlims.max() * 2)

        ax = plt.subplot(nrows, ncols, iplot)
        ax2 = ax.inset_axes(
            [0.6, 0.78, 0.4, 0.22], xscale="log", xlim=xlims, ylabel=inset_ylabel,
        )
        plt.sca(ax)
        sel1 = preds_df[base_target_column] < base_target_thr
        sel2 = (~sel1) & preds_df[extrap_target_column].isna()
        if triangle_clipped:
            xx = _f(preds_df[col].loc[sel1])
            yy = np.log10((preds_df[base_target_column].clip(lower=base_target_clip_low)).loc[sel1])
            clip_sel = np.isclose(yy, np.log10(base_target_clip_low))
            plt.errorbar(xx[~clip_sel], y=yy[~clip_sel], color='k', fmt='o')
            plt.errorbar(xx[clip_sel], y=yy[clip_sel], color='k', fmt='v')
        else:
            plt.errorbar(
                _f(preds_df[col].loc[sel1]),
                y=np.log10((preds_df[base_target_column].clip(lower=base_target_clip_low)).loc[sel1]),
                color='k', fmt='o',
            )
        plt.plot(
            _f(preds_df[col].loc[sel2]),
            np.log10((preds_df[base_target_column].clip(lower=base_target_clip_low)).loc[sel2]),
            'o',
            markeredgecolor='#99999977', markerfacecolor='#00000000',
        )

        for _, row in preds_df.iterrows():
            if np.isnan(row[extrap_target_column]): continue
            plotted_objects = ax2.errorbar(
                x=_f(row[col]),
                y=row[extrap_target_column],
                yerr=row[f"{extrap_target_column}_err"],
                fmt='o',
            )
            plt.sca(ax)
            color = plotted_objects.get_children()[0].get_color()
            plt.errorbar(
                x=_f(row[col]),
                y=np.log10(row[base_target_column]),
                fmt='o',
                markeredgecolor=color,
                markerfacecolor=color if main_fill_colored else '#00000000',
            )


        plt.xscale("log")
        if xlabel_map is not None:
            plt.xlabel(xlabel_map[col])
        else:
            plt.xlabel(f"1 / {col}" if ("barrier" in col) and transform_barriers else col)
        plt.xlim(*xlims)
        plt.ylabel(main_ylabel)
        plt.ylim(*base_target_lims)

    plt.tight_layout()
    return fig

class MetricsParams:
    extrapolated_target: str = "log10condNE300"
    extrapolated_target_clip_low: float = 1e-6
    base_target: str = "condNE1000"
    base_target_clip_low: float = 1e-7
    base_target_min_positive: float = 1e-2
    weight_col: str = "w_extrap"

class PairwiseComparisonMetrics(MetricsParams):
    @staticmethod
    def _prob_a_gt_b(*, a, a_err, b, b_err):
        diff = (a - b) / np.sqrt(a_err**2 + b_err**2)
        return np.math.erfc(-diff) / 2

    def prob_row_a_better_than_b(
        self, a: pd.Series, b: pd.Series,
    ) -> float:
        # case 1: both successfully extrapolated
        if not (np.isnan(a[self.extrapolated_target]) or np.isnan(b[self.extrapolated_target])):
            # then just compare gaussians:
            return PairwiseComparisonMetrics._prob_a_gt_b(
                a=a[self.extrapolated_target], a_err=a[f"{self.extrapolated_target}_err"],
                b=b[self.extrapolated_target], b_err=b[f"{self.extrapolated_target}_err"],
            )

        D_a = np.clip(a[self.base_target], a_min=self.base_target_clip_low, a_max=None)
        D_b = np.clip(b[self.base_target], a_min=self.base_target_clip_low, a_max=None)

        # case 2: at least one of them not extrapolated and at least one
        # can potantially be good
        if max(D_a, D_b) > self.base_target_min_positive:
            # only if they differ by at least three orders we compare gaussians
            if np.abs(np.log10(D_a / D_b)) > 3:
                return PairwiseComparisonMetrics._prob_a_gt_b(
                    a=D_a, a_err=a[f"{self.base_target}_err"],
                    b=D_b, b_err=b[f"{self.base_target}_err"],
                )

        # case 3: they're probably both bad, or at least uncomparable, so we don't care
        return float("NaN")

    def comparison_weighted(self, row_a: pd.Series, row_b: pd.Series) -> float:
        return (self.prob_row_a_better_than_b(row_a, row_b) * 2 - 1) * row_a[self.weight_col] * row_b[self.weight_col]

    def comparison_unweighted(self, row_a: pd.Series, row_b: pd.Series) -> float:
        return self.prob_row_a_better_than_b(row_a, row_b) * 2 - 1

    def select_and_plot_best_features(
        self,
        comparison: Literal["weighted", "unweighted"],
        preds_df: pd.DataFrame,
        features_meta_info: pd.DataFrame,
        print_top: int = 20,
        visualizations: bool = True,
    ) -> pd.Series:
        if comparison == "weighted":
            comparison_func = self.comparison_weighted
        elif comparison == "unweighted":
            comparison_func = self.comparison_unweighted
        else:
            raise NotImplementedError(comparison)

        preds_df = preds_df.copy()
        cols_to_test = list(features_meta_info.index)
        inv_cols = list(features_meta_info.query("weighted_direction < 0").index)
        preds_df[inv_cols] = -preds_df[inv_cols]

        DD_a, DD_b, aVSb = zip(*[
            (row_a, row_b, comparison_func(row_a, row_b))
            for _, row_a in preds_df.iterrows() for _, row_b in preds_df.iterrows()
        ])
        DD_a = pd.DataFrame(DD_a).reset_index(drop=True)
        DD_b = pd.DataFrame(DD_b).reset_index(drop=True)
        aVSb = np.array(aVSb)

        score_pos = (
            ((DD_a[cols_to_test] - DD_b[cols_to_test]) * aVSb[:, None] > 0) * np.abs(aVSb[:, None])
        ).sum(axis=0)

        if visualizations:
            # visualize weights
            plt.figure()
            plt.scatter(
                DD_a[self.base_target].clip(lower=self.base_target_clip_low),
                DD_b[self.base_target].clip(lower=self.base_target_clip_low),
                c=aVSb, cmap='seismic',
            )
            plt.colorbar()
            plt.xscale("log")
            plt.yscale("log")

        top_features = (
            score_pos / np.abs(aVSb[~np.isnan(aVSb)]).sum()
        ).abs().sort_values(ascending=False)
        if print_top > 0:
            print(top_features.iloc[:print_top])

        # visualize top features
        preds_df[inv_cols] = -preds_df[inv_cols]  # invert barriers back
        if visualizations:
            plot_features_Kahle2020(top_features.index[:12], preds_df=preds_df)
        return top_features.sort_index()


class ROClikeComparisonMetrics(MetricsParams):
    def eval_features(
        self,
        preds_df: pd.DataFrame,
        features_meta_info: pd.DataFrame,
        positive_thr: float = 1e-1,
        negative_thr: float = 1e-2,
        num_negatives_max: float = 5.0,
        num_bootstrap_samples: int = 100,
        rng_seed: int = 42,
        weight_validation_plots: bool = False,
        positive_on_extrap300: bool = False,
        dataset_type: Literal["Kahle2020", "experimental"] = "Kahle2020",
        experimental_error_relative: float = 0.5,
    ) -> pd.DataFrame:
        preds_df = preds_df.copy()
        erfc = np.vectorize(np.math.erfc)

        if dataset_type == "Kahle2020":
            positive_prob_src = self.extrapolated_target if positive_on_extrap300 else self.base_target
            negative_prob_src = self.base_target
        elif dataset_type == "experimental":
            positive_prob_src = negative_prob_src = "sigma_S_cm"
            preds_df["sigma_S_cm"] = preds_df["sigma_S_cm"].replace("<1E-10", "1e-10").astype(float)
            preds_df["sigma_S_cm_err"] = preds_df["sigma_S_cm"] * experimental_error_relative
        else:
            raise NotImplementedError(dataset_type)

        sample_weights = 1.0
        if "sample_weight" in preds_df.columns:
            print("Found sample weights!")
            sample_weights = preds_df["sample_weight"]
        preds_df["positive_prob"] = sample_weights * erfc(
            (positive_thr - preds_df[positive_prob_src].fillna(-99999)) / preds_df[f"{positive_prob_src}_err"].fillna(1.0)
        ) / 2
        preds_df["negative_prob"] = sample_weights * erfc(
            -(negative_thr - preds_df[negative_prob_src]) / preds_df[f"{negative_prob_src}_err"]
        ) / 2

        if weight_validation_plots:
            for w_col in ["positive_prob", "negative_prob"]:
                plt.figure()
                plt.scatter(
                    preds_df[w_col],
                    preds_df[negative_prob_src].clip(lower=1e-10),
                    c=(
                        np.where(
                            preds_df[self.extrapolated_target].isna(),
                            self.extrapolated_target_clip_low,
                            10**preds_df[self.extrapolated_target]
                        ) if dataset_type == "Kahle2020" else None
                    ),
                    norm=mpl.colors.LogNorm() if dataset_type == "Kahle2020" else None,
                )
                if dataset_type == "Kahle2020":
                    plt.colorbar()
                plt.yscale("log")


        rng = np.random.default_rng(seed=rng_seed)
        bs_ids = rng.choice(
            len(preds_df),
            size=(num_bootstrap_samples, len(preds_df)),
            replace=True,
        )
        bs_preds_dfs = [preds_df.iloc[bs_ids_i] for bs_ids_i in bs_ids]
        bs_scores_df = pd.concat([
            ROClikeComparisonMetrics._eval_features_central(
                preds_df=pred_df,
                num_negatives_max=num_negatives_max,
                features_meta_info=features_meta_info,
            )["score"] for pred_df in tqdm(bs_preds_dfs)
        ], axis=1)

        scores_df = ROClikeComparisonMetrics._eval_features_central(
            preds_df=preds_df,
            num_negatives_max=num_negatives_max,
            features_meta_info=features_meta_info,
        )

        scores_df["score_bs_q16"] = bs_scores_df.quantile(0.16, axis=1)
        scores_df["score_bs_q50"] = bs_scores_df.quantile(0.50, axis=1)
        scores_df["score_bs_q84"] = bs_scores_df.quantile(0.84, axis=1)
        return scores_df

    @staticmethod
    def _eval_features_central(
        preds_df: pd.DataFrame,
        num_negatives_max: float,
        features_meta_info: pd.DataFrame,
    ) -> pd.DataFrame:
        preds_df = preds_df.copy()

        def calculate_roclike_curve(
            feature_name: str,
            larger_better: bool,
            prob_zero_thr: float = 1e-3,
        ) -> Tuple[np.ndarray, np.ndarray]:
            sorted_df = preds_df.sort_values(feature_name, ascending=not larger_better)
            p_pos = sorted_df["positive_prob"].to_numpy()
            p_neg = sorted_df["negative_prob"].to_numpy()

            assert (p_pos >= 0).all()
            assert (p_neg >= 0).all()

            nonzero_mask = (p_pos > prob_zero_thr) | (p_neg > prob_zero_thr)
            p_pos = np.concatenate([[0.0], p_pos[nonzero_mask]])
            p_neg = np.concatenate([[0.0], p_neg[nonzero_mask]])

            return p_pos.cumsum(), p_neg.cumsum()

        def integrate_roc(
            w_pos: np.ndarray,
            w_neg: np.ndarray,
            w_neg_max: float,
            validate_with_ref: bool = False,
        ) -> float:
            assert w_neg_max > 0
            assert w_pos.ndim == 1
            assert w_pos.shape == w_neg.shape
            assert len(w_pos) >= 2
            assert w_pos[0] == w_neg[0] == 0.0

            dw_pos = np.diff(w_pos)
            dw_neg = np.diff(w_neg)
            assert (dw_pos >= 0).all()
            assert (dw_neg >= 0).all()
            assert ((dw_pos > 0) | (dw_neg > 0)).all()
            del dw_pos

            if validate_with_ref:
                def slow_ref_calculation(pp, nn):
                    integral = 0.0

                    for i in range(len(pp) - 1):
                        l, r = nn[i: i + 2]
                        b, t = pp[i: i + 2]
                        es = False
                        if r >= w_neg_max:
                            scale = (w_neg_max - l) / (r - l)
                            r = w_neg_max
                            t = b + scale * (t - b)
                            es = True
                        integral += (r - l) * (t + b) / 2
                        if es: break
                    return integral
                slow_ref_integral = slow_ref_calculation(w_pos, w_neg)

            sel = np.concatenate([[True], w_neg < w_neg_max])[:-1]
            assert sel[1]

            (w_pos, w_neg, dw_neg) = (w_pos[sel], w_neg[sel], dw_neg[sel[1:]])
            if w_neg[-1] > w_neg_max:
                assert w_neg[-2] < w_neg_max
                scaling = (w_neg_max - w_neg[-2]) / dw_neg[-1]
                dw_neg[-1] *= scaling
                w_neg[-1] = w_neg[-2] + dw_neg[-1]
                assert np.isclose(w_neg[-1], w_neg_max)
                w_pos[-1] = w_pos[-2] + scaling * (w_pos[-1] - w_pos[-2])

            integral = ((w_pos[1:] + w_pos[:-1]) * dw_neg / 2).sum()
            if validate_with_ref:
                assert np.isclose(integral, slow_ref_integral), (integral, slow_ref_integral)
            return integral

        result_data = []
        for col, meta in features_meta_info.iterrows():
            roclike = calculate_roclike_curve(col, larger_better=meta["weighted_direction"] > 0)
            score = integrate_roc(*roclike, w_neg_max=num_negatives_max)
            result_data.append(dict(
                score=score,
                roclike=roclike,
            ))

        return pd.DataFrame(
            result_data, index=features_meta_info.index,
        ).sort_index()

    @staticmethod
    def plot_scores(
        features: List[str],
        scores_to_plot: pd.DataFrame,
        scores_to_color: pd.DataFrame,
        label_main: str = "condNE1000 score (bs conf. intervals)",
        label_color: str = "extrapolated condNE300",
    ):
        scores_to_plot = scores_to_plot.copy().loc[features]
        scores_to_color = scores_to_color.copy().loc[features]

        fig = plt.figure(figsize=(17, 6))
        base = scores_to_plot["score_bs_q16"]
        height = scores_to_plot["score_bs_q84"] - scores_to_plot["score_bs_q16"]

        gs = mpl.gridspec.GridSpec(3, 2, width_ratios=[4, 1])

        ax_main = fig.add_subplot(gs[:, 0])
        ax_cb = [fig.add_subplot(gs[i, 1]) for i in range(3)]

        for i_bit in range(3):
            color_score_col = ["score_bs_q16", "score_bs_q50", "score_bs_q84"][i_bit]
            color_src = scores_to_color[color_score_col]
            mapper = mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=0, vmax=color_src.max()),
            )
            plt.sca(ax_main)
            plt.bar(
                np.arange(len(features)),
                height=height / 3,
                bottom=base + height * i_bit / 3,
                color=mapper.to_rgba(color_src),
            )
            for i, val in enumerate(scores_to_plot["score_bs_q50"]):
                plt.plot([i - 0.3, i + 0.3], [val] * 2, color='k')
            plt.colorbar(mapper, cax=ax_cb[::-1][i_bit], label=f"{label_color} {color_score_col}", orientation='horizontal')

        plt.xticks(np.arange(len(features)), features, rotation=90);
        plt.ylim(bottom=0.0);
        plt.ylabel(label_main)
        plt.tight_layout();

def plot_feature_roclikes(
    feature_scores: pd.DataFrame,
    factor: float = 3,
    min_y: float = 10,
    max_y: float = 30,
    num_negatives_max: float = 5.0,
    title: str = "",
    logx: bool = False,
    min_x: Optional[float] = None,
):
    plt.figure(figsize=(10 * factor, 10 * factor), dpi=100)
    width = len(feature_scores) / 2 + 1
    for feature, f_info in feature_scores.iterrows():
        positive, negative = f_info["roclike"]
        plt.plot(
            negative, positive,
            label=feature,
            linewidth=width,
            linestyle="dashed",
            alpha=0.6,
        )
        width -= 0.5
    plt.xlabel("num negatives")
    plt.ylabel("num positives")
    plt.xlim(0.0, num_negatives_max)
    plt.ylim(min_y, max_y)
    plt.legend(fontsize=7)
    if title:
        plt.title(title)
    if logx:
        plt.xscale("log")
    if min_x is not None:
        plt.xlim(left=min_x)