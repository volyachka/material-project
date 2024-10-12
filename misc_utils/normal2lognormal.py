from typing import Tuple, Literal

from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
import sympy as sp


def _KL_lognorm_norm_ref(mu_lognorm, sigma_lognorm, mu_norm, sigma_norm):
    """
    KL-divergence between log-normal and normal distributions

    Parameters
    ----------
    mu_lognorm, sigma_lognorm
        parameters of the log-normal distribution
    mu_norm, sigma_norm
        mean and stddev, respectively, for the normal distribution

    Returns
    -------
    KL_value
        value of the KL divergence
    """
    return (
        -1.0 / 2 + np.log(sigma_norm / sigma_lognorm) - mu_lognorm
        + 1.0 / (2 * sigma_norm**2) * (
            (mu_norm - np.exp(mu_lognorm + sigma_lognorm**2 / 2))**2
            + np.exp(2 * mu_lognorm + sigma_lognorm**2) * (np.exp(sigma_lognorm**2) - 1)
        )
    )

@tf.function
def KL_lognorm_norm(
    expmu_lognorm: tf.Tensor,
    expsigma2ovr2_lognorm: tf.Tensor,
    mu_norm: tf.Tensor,
    sigma_norm: tf.Tensor,
):
    """
    KL-divergence between log-normal and normal distributions

    Parameters
    ----------
    expmu_lognorm
        exponent of the mu (location) parameter of the log-normal distribution
    expsigma2ovr2_lognorm
        value of exp(sigma**2 / 2), where sigma is the width parameter of the log-normal
        distribution
    mu_norm, sigma_norm
        mean and stddev, respectively, for the normal distribution

    Returns
    -------
    KL_value
        value of the KL divergence
    """
    s = tf.sqrt(2 * tf.math.log(expsigma2ovr2_lognorm))
    return (
        -1.0 / 2 + tf.math.log(sigma_norm / s / expmu_lognorm)
        + 1.0 / (2 * sigma_norm**2) * (
            mu_norm**2
            - 2 * mu_norm * expmu_lognorm * expsigma2ovr2_lognorm
            + expmu_lognorm**2 * expsigma2ovr2_lognorm**4
        )
    )

def fit_lognorm2norm(
    mu: np.ndarray,
    sigma: np.ndarray,
    kl_abs_accuracy: float = 1e-6,
    accuracy_window_size: int = 5,
    max_steps: int = 1000,
    learning_rate: float = 0.001,
    device: str = "CPU",
    dtype: Literal["float32", "float64"] = "float64",
    debug: bool = False,
    show_progress: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Given the parameters of a normal distribution N, find a log-normal distribution LN
    that minimizes KL(LN||N).

    Parameters
    ----------
    mu: np.ndarray
        mean of the input normal distribution
    sigma: np.ndarray
        stddev of the input normal distribution
    kl_abs_accuracy: float
        KL precision (default = 1e-6)
    accuracy_window_size: int
        Number of latest update steps to track KL values for (default = 5). The largest deviation
        from mean over this window is compared with `kl_abs_accuracy` to define the successful
        optimization.
    max_steps: int
        Max number of optimization steps (default = 1000). Failing to converge within this many
        steps will raise a RuntimeError exception.
    learning_rate: float
        Gradient descent step size (default = 0.001)
    device: str
        Tensorflow device to use (default = "CPU")
    dtype: Literal["float32", "float64"]
        Tensorflow floating point data type (default = "float64")
    debug: bool
        Whether to make dubug printouts (default = False)
    show_progress
        Whether to show progress bar (default = False)

    Returns
    -------
    (mu_lognorm, sigma_lognorm), KL
        Fitted parameters of the log-normal distribution and the found KL(LN||N)
    """
    assert mu.ndim == 1
    assert mu.shape == sigma.shape
    assert (sigma > 0).all()
    scales = sigma.copy()
    sigma = np.ones_like(sigma)
    mu = mu / scales

    _prog = tqdm if show_progress else (lambda x, **args: x)

    best_m, best_s, best_KL = np.array([
        _make_guess_bruteforce(mu_i, sigma_i)
        for mu_i, sigma_i in zip(_prog(mu, desc="Making initial guess"), sigma)
    ]).T

    _f = lambda x: tf.convert_to_tensor(x, dtype=dtype)

    with tf.device(device):
        musigma = _f([mu, sigma])
        mlogs = tf.Variable(
            [
                best_m,
                np.log(best_s),
            ], dtype=dtype,
        )
        assert mlogs.shape == musigma.shape
        MS = tf.stack([tf.exp(mlogs[0]), tf.exp(tf.exp(2 * mlogs[1]) / 2)], axis=0)
        np.testing.assert_allclose(
            KL_lognorm_norm(*MS, *musigma).numpy(),
            best_KL,
        )

        opt = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

        memorized_losses = [None] * accuracy_window_size
        err = False
        for i_step in _prog(range(max_steps), desc="refining"):
            if debug:
                print(f"\n\n------ step: {i_step + 1} ------\n")
            with tf.GradientTape() as t:
                MS = tf.stack([tf.exp(mlogs[0]), tf.exp(tf.exp(2 * mlogs[1]) / 2)], axis=0)
                loss = KL_lognorm_norm(*MS, *musigma)
                loss_agg = tf.reduce_mean(loss)

            memorized_losses = memorized_losses[1:] + [loss.numpy()]
            grads = t.gradient(loss_agg, mlogs)
            if debug:
                print("grads:")
                print(grads.numpy())
                print("\nm, log(s)")
                print(mlogs.numpy())
            opt.apply_gradients([(grads, mlogs)])
            if i_step + 1 >= accuracy_window_size:
                memorized_losses_arr = np.array(memorized_losses)
                error_per_component = np.abs(
                    memorized_losses_arr - memorized_losses_arr.mean(axis=0, keepdims=True)
                ).max(axis=0)
                error = error_per_component.max()

                min_kl_per_component = memorized_losses_arr.min(axis=0)
                rel_error = (error_per_component / min_kl_per_component).max()
                if debug:
                    print(f"\n{'rel err':8} - {'abs err':8}")
                    print(f"{rel_error:8.3e} - {error:8.3e}")

                if error <= kl_abs_accuracy:
                    if debug:
                        print("\n ---- - - - success! - - - ----")
                    break
        else:
            if debug:
                print("\n == == = == = didn't converge = = == = = == ")
            err = True

    fitted_expmu_lognorm, fitted_expsigma2ovr2_lognorm = MS.numpy()
    fitted_mu_lognorm = np.log(fitted_expmu_lognorm) + np.log(scales)
    fitted_sigma_lognorm = np.sqrt(2 * np.log(fitted_expsigma2ovr2_lognorm))

    assert fitted_mu_lognorm.shape == mu.shape
    assert fitted_sigma_lognorm.shape == sigma.shape

    np.testing.assert_allclose(
        memorized_losses[-1], _KL_lognorm_norm_ref(fitted_mu_lognorm, fitted_sigma_lognorm, mu * scales, scales)
    )
    assert (memorized_losses[-1] <= best_KL).all()

    if err:
        if debug:
            print("errorring")
            print(f"{mu=}")
            print(f"{sigma=}")
            print(f"{fitted_expmu_lognorm=}")
            print(f"{fitted_expsigma2ovr2_lognorm=}")
        raise RuntimeError("Could not converge", musigma.numpy(), memorized_losses)

    return (fitted_mu_lognorm, fitted_sigma_lognorm), memorized_losses[-1]

def _def_symbols():
    M, S, mu = sp.symbols("M S mu")

    KL_p1 = sp.log(1 / (M * sp.sqrt(sp.log(S))))
    KL_p2 = ((M * S - mu)**2 + (M * S)**2 * (S**2 - 1)) / 2
    KL = KL_p1 + KL_p2 - sp.log(2 * sp.exp(1)) / 2

    M_solution = (mu + sp.sqrt(4 * S**2 + mu**2)) / (2 * S**3)

    det_KL = sp.simplify(sp.simplify(
        sp.simplify(sp.diff(KL, M, M) * sp.diff(KL, S, S))
        - sp.simplify(sp.diff(KL, S, M)**2)
    ).subs({M: M_solution}))
    eq_S = sp.simplify(sp.diff(KL, S).subs({M: M_solution}))

    return dict(
        M=M, S=S, mu=mu, KL=KL, det_KL=det_KL, eq_S=eq_S, M_solution=M_solution
    )

_SYMBOLS = _def_symbols()

def _make_guess_bruteforce(
    mu_norm: float,
    sigma_norm: float,
    log10_Sm1_min: float = -15.0,
    log10_Sm1_max: float = 20.0,
    nsteps: int = 10_000,
    make_val_plot: bool = False,
):
    assert sigma_norm > 0

    mu_norm_scaled = mu_norm / sigma_norm
    assert mu_norm_scaled <= 1000.0, "Normal distribution is too narrow, current search method may get unstable"
    assert log10_Sm1_min >= -15

    M, S, mu, KL, det_KL, eq_S, M_solution = [_SYMBOLS[s] for s in "M S mu KL det_KL eq_S M_solution".split()]

    det_KL = sp.lambdify([S], det_KL.subs({mu: mu_norm_scaled}))
    eq_S = sp.lambdify([S], eq_S.subs({mu: mu_norm_scaled}))

    Svalsm1 = 10**np.linspace(
        log10_Sm1_min, log10_Sm1_max, nsteps,
    )

    eq_S_vals = eq_S(Svalsm1 + 1.0)
    det_KL_vals = det_KL(Svalsm1 + 1.0)

    if make_val_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(
            Svalsm1, np.abs(eq_S_vals)
        )
        plt.xscale("log")
        plt.yscale("log")

    i_best = np.abs(np.diff(np.log(np.abs(eq_S_vals)))).argmax()
    assert det_KL_vals[i_best] > 0
    if make_val_plot:
        plt.plot([Svalsm1[i_best]] * 2, [1-12, 1e12])

    S_best = Svalsm1[i_best] + 1.0
    M_best = float(M_solution.subs({S: S_best, mu: mu_norm_scaled}).evalf())
    KL_best = float(KL.subs({M: M_best, S: S_best, mu: mu_norm_scaled}).evalf())

    mu_lognorm_best = np.log(M_best) + np.log(sigma_norm)
    sigma_lognorm_best = np.sqrt(2 * np.log(S_best))
    return mu_lognorm_best, sigma_lognorm_best, KL_best
