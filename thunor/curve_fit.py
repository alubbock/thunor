import numpy as np
import scipy.optimize
import scipy.stats
from abc import abstractmethod
import pandas as pd
import warnings

PARAM_EQUAL_ATOL = 1e-16
PARAM_EQUAL_RTOL = 1e-12
# Curve fitting E0 tolerance parameters
E0_TEST_MIN_CTRLS_STD_DEV = 5  # min # controls to use std. dev.
E0_TEST_MULTIPLIER = 1.2
# Value to divide minimum dose by to get "control dose" for visualization
# and dip fits
CTRL_DOSE_DIVISOR = 10.0


class ValueWarning(UserWarning):
    pass


class AUCFitWarning(ValueWarning):
    pass


class AAFitWarning(ValueWarning):
    pass


class DrugCombosWarning(UserWarning):
    """
    Warning issued when drug combination wells are skipped during fitting

    :func:`fit_params_minimal` currently fits single-drug dose-response curves
    only.  Combination wells (where the ``drug`` tuple has length > 1) are
    filtered out and this warning is issued.  Future versions will support
    combination fitting via a dedicated code path; the skip-and-warn behaviour
    is intentional and will not change when that support lands.
    """

    pass


class HillCurve(object):
    """Base class defining Hill/log-logistic curve functionality"""

    fit_bounds = (-np.inf, np.inf)
    curve_fit_kwargs = {}
    curve_fit_kwargs_log = {}
    null_response_fn = np.mean
    max_fit_evals = None

    def __init__(self, popt):
        self.popt = popt

    @classmethod
    @abstractmethod
    def fit_fn(cls, x, *params):
        pass

    def fit(self, x):
        return self.fit_fn(x, *self.popt)

    def fit_rel(self, x):
        return None

    @classmethod
    @abstractmethod
    def initial_guess(cls, x, y):
        pass

    @abstractmethod
    def divisor(self):
        pass

    @abstractmethod
    def ec50(self):
        pass

    @abstractmethod
    def e0(self):
        pass

    @abstractmethod
    def emax(self):
        pass

    @abstractmethod
    def hill_slope(self):
        pass


class HillCurveNull(HillCurve):
    @classmethod
    def fit_fn(cls, x, ymean):
        return ymean

    def fit(self, x):
        return self.fit_fn(x, self.popt)

    @classmethod
    def initial_guess(cls, x, y):
        return np.mean(y)

    @property
    def divisor(self):
        return self.popt

    def ic(self, ic_num=50):
        return None

    def ec(self, ec_num=50):
        return None

    @property
    def ec50(self):
        return None

    @property
    def e0(self):
        return self.popt

    @property
    def emax(self):
        return self.popt

    @property
    def hill_slope(self):
        return None

    def auc(self, *args, **kwargs):
        return 0.0

    def aa(self, *args, **kwargs):
        return 0.0


class HillCurveLL4(HillCurve):
    max_fit_evals = 10000

    def __init__(self, popt):
        super(HillCurveLL4, self).__init__(popt)
        self._popt_rel = None

    @classmethod
    def undo_dose_scale(cls, popt, dose_scale):
        """Undo dose scaling on linear-space popt (EC50 is at index 3)."""
        popt = np.array(popt, dtype=float)
        popt[3] *= dose_scale
        return popt

    @classmethod
    def fit_fn(cls, x, b, c, d, e):
        """
        Four parameter log-logistic function ("Hill curve")

        Parameters
        ----------
        x: np.ndarray
            One-dimensional array of "x" values
        b: float
            Hill slope
        c: float
            Maximum response (lower plateau)
        d: float
            Minimum response (upper plateau)
        e: float
            EC50 value

        Returns
        -------
        np.ndarray
            Array of "y" values using the supplied curve fit parameters on "x"
        """
        return c + (d - c) / (1 + np.exp(b * (np.log(x) - np.log(e))))

    @classmethod
    def fit_fn_log(cls, x, b, c, d, log_e):
        """LL4 Hill curve with log-transformed EC50 (log_e = log(e)).

        Using log(EC50) as the optimisation parameter removes the positivity
        constraint on e, enabling the faster Levenberg-Marquardt solver.
        """
        return c + (d - c) / (1.0 + np.exp(b * (np.log(x) - log_e)))

    @classmethod
    def jac_fn_log(cls, x, b, c, d, log_e):
        """Analytic Jacobian of LL4 w.r.t. (b, c, d, log_e).

        Avoids ~4 finite-difference evaluations per optimiser step.
        """
        log_x = np.log(x)
        s = np.exp(b * (log_x - log_e))
        denom = 1.0 + s
        denom_sq = denom * denom

        j_b = -(d - c) * s * (log_x - log_e) / denom_sq
        j_c = s / denom  # = 1 - 1/denom
        j_d = 1.0 / denom
        j_log_e = (d - c) * b * s / denom_sq

        jac = np.column_stack((j_b, j_c, j_d, j_log_e))
        jac[np.isnan(jac)] = 0.0
        return jac

    @classmethod
    def initial_guess_log(cls, x, y, dose_scale=1.0):
        """Initial guess in log-EC50 space, accounting for dose scaling."""
        b, c, d, e = cls.initial_guess(x, y)
        with np.errstate(divide='ignore'):
            log_e = np.log(e / dose_scale)
        return b, c, d, log_e

    @classmethod
    def transform_popt_from_log(cls, popt, dose_scale=1.0):
        """Back-transform popt from log-EC50 space to linear EC50."""
        popt = np.array(popt, dtype=float)
        popt[3] = np.exp(popt[3]) * dose_scale
        return popt

    # Bounds in log-EC50 space: no bound on log_e (was e > 0)
    fit_bounds_log = (-np.inf, np.inf)

    @classmethod
    def initial_guess(cls, x, y):
        """
        Heuristic function for initial fit values

        Uses the approach followed by R's drc library:
        https://cran.r-project.org/web/packages/drc/index.html

        Parameters
        ----------
        x: np.ndarray
            Array of "x" (dose) values
        y: np.ndarray
            Array of "y" (response) values

        Returns
        -------
        list
            Four-valued list corresponding to initial estimates of the
            parameters defined in the :func:`ll4` function.
        """
        c_val, d_val = _find_cd_ll4(y)
        b_val, e_val = _find_be_ll4(x, y, c_val, d_val)

        return b_val, c_val, d_val, e_val

    @property
    def ec50(self):
        return self.popt[3]

    @property
    def e0(self):
        return self.popt[2]

    @property
    def emax(self):
        return self.popt[1]

    @property
    def hill_slope(self):
        return self.popt[0]

    def ic(self, ic_num=50):
        """
        Find the inhibitory concentration value (e.g. IC50)

        Parameters
        ----------
        ic_num: int
            IC number between 0 and 100 (response level)

        Returns
        -------
        float
            Inhibitory concentration value for requested response value
        """
        emax = self.emax
        if not isinstance(emax, float):
            return None
        e0 = self.e0
        if emax > e0:
            emax, e0 = e0, emax

        ic_frac = ic_num / 100.0

        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            icN = self.ec50 * (ic_frac / (1 - ic_frac - (emax / e0))) ** (
                1 / self.hill_slope
            )

        # Overflow will lead to -inf, which we deal with here
        if np.isnan(icN) or np.isinf(icN):
            icN = None

        return icN

    def ec(self, ec_num=50):
        """
        Find the effective concentration value (e.g. IC50)

        Parameters
        ----------
        ec_num: int
            EC number between 0 and 100 (response level)

        Returns
        -------
        float
            Effective concentration value for requested response value
        """
        if ec_num >= 100:
            return None

        ec_frac = ec_num / 100.0

        return self.ec50 * (ec_frac / (1 - ec_frac)) ** (1 / self.hill_slope)

    def auc(self, min_conc):
        """
        Find the area under the curve

        Parameters
        ----------
        min_conc: float
            Minimum concentration to consider for fitting the curve

        Returns
        -------
        float or None
            Area under the curve (AUC) value, or ``None`` for stimulatory
            responses (Emax > E0) which are not yet supported.
        """
        emax = self.emax
        if not isinstance(emax, float):
            return None
        e0 = self.e0
        if emax > e0:
            # TODO: Calculate AUC for ascending curves
            return None

        with np.errstate(divide='ignore', invalid='ignore'):
            min_conc_hill = min_conc**self.hill_slope
            result = (
                np.log10((self.ec50**self.hill_slope + min_conc_hill) / min_conc_hill)
                / self.hill_slope
            ) * ((e0 - emax) / e0)
        if np.isnan(result) or np.isinf(result):
            return None
        return result

    def aa(self, min_conc, max_conc):
        """
        Find the activity area (area over the curve)

        Parameters
        ----------
        min_conc: float
            Minimum concentration to consider for fitting the curve
        max_conc: float
            Maximum concentration to consider for fitting the curve

        Returns
        -------
        float or None
            Activity area value, or ``None`` for stimulatory responses
            (Emax > E0) which are not yet supported.
        """
        emax = self.emax
        if not isinstance(emax, float):
            return None
        e0 = self.e0
        if emax > e0:
            # TODO: Calculate AA for ascending curves
            return None

        hill = self.hill_slope
        ec50 = self.ec50
        log_ec50 = np.log10(ec50)

        # Use log-space arithmetic to avoid overflow from ec50**hill or conc**hill.
        # log10(ec50^h + x^h) = h*log10(ec50) + log10(1 + (x/ec50)^h)
        #                      = h*log10(ec50) + log10(1 + 10^(h*(log10(x)-log10(ec50))))
        # The ec50^h terms cancel in the difference, leaving:
        # aa = [log10(1 + 10^r_max) - log10(1 + 10^r_min)] / hill * (e0-emax)/e0
        # where r = hill * (log10(x) - log10(ec50))
        def _log10_1p_pow10(r):
            # log10(1 + 10^r), numerically stable for large |r|
            if r > 100.0:
                return r  # 10^r >> 1
            return np.log10(1.0 + 10.0**r)

        with np.errstate(divide='ignore', invalid='ignore'):
            r_max = hill * (np.log10(max_conc) - log_ec50)
            r_min = hill * (np.log10(min_conc) - log_ec50)
            result = (
                (_log10_1p_pow10(r_max) - _log10_1p_pow10(r_min))
                / hill
                * ((e0 - emax) / e0)
            )
        if np.isnan(result) or np.isinf(result):
            return None
        return result

    @property
    def divisor(self):
        return max(self.emax, self.e0)

    @property
    def popt_rel(self):
        if self._popt_rel is None:
            self._popt_rel = self.popt.copy()
            self._popt_rel[2] /= self.divisor
            self._popt_rel[1] /= self.divisor
        return self._popt_rel

    def fit_rel(self, x):
        return self.fit_fn(x, *self.popt_rel)


class HillCurveLL3u(HillCurveLL4):
    """Three parameter log logistic curve, for viability data"""

    # Constrain 0<=emax<=1, Hill slope +ve
    fit_bounds = ((0.0, 0.0, -np.inf), (np.inf, 1.0, np.inf))
    curve_fit_kwargs = {'method': 'dogbox'}
    # In log-EC50 space: bounds only on b (slope ≥ 0) and c (0 ≤ emax ≤ 1);
    # no bound needed on log_e since exp(log_e) > 0 always
    fit_bounds_log = ((0.0, 0.0, -np.inf), (np.inf, 1.0, np.inf))
    curve_fit_kwargs_log = {'method': 'dogbox', 'x_scale': 'jac'}
    max_fit_evals = None

    @staticmethod
    def null_response_fn(_):
        return np.float64(1.0)

    @classmethod
    def fit_fn(cls, x, b, c, e):
        """
        Three parameter log-logistic function ("Hill curve")

        Parameters
        ----------
        x: np.ndarray
            One-dimensional array of "x" values
        b: float
            Hill slope
        c: float
            Maximum response (lower plateau)
        e: float
            EC50 value

        Returns
        -------
        np.ndarray
            Array of "y" values using the supplied curve fit parameters on "x"
        """
        return super(HillCurveLL3u, cls).fit_fn(x, b, c, 1.0, e)

    @classmethod
    def fit_fn_log(cls, x, b, c, log_e):
        """LL3u Hill curve with log-transformed EC50."""
        return c + (1.0 - c) / (1.0 + np.exp(b * (np.log(x) - log_e)))

    @classmethod
    def jac_fn_log(cls, x, b, c, log_e):
        """Analytic Jacobian of LL3u w.r.t. (b, c, log_e)."""
        log_x = np.log(x)
        s = np.exp(b * (log_x - log_e))
        denom = 1.0 + s
        denom_sq = denom * denom

        j_b = -(1.0 - c) * s * (log_x - log_e) / denom_sq
        j_c = s / denom
        j_log_e = (1.0 - c) * b * s / denom_sq

        jac = np.column_stack((j_b, j_c, j_log_e))
        jac[np.isnan(jac)] = 0.0
        return jac

    @classmethod
    def initial_guess(cls, x, y):
        hill, emax, _, ec50 = super().initial_guess(x, y)
        if emax < 0.0:
            emax = 0.0
        elif emax > 1.0:
            emax = 1.0
        return hill, emax, ec50

    @classmethod
    def initial_guess_log(cls, x, y, dose_scale=1.0):
        b, c, e = cls.initial_guess(x, y)
        with np.errstate(divide='ignore'):
            log_e = np.log(e / dose_scale)
        return b, c, log_e

    @classmethod
    def transform_popt_from_log(cls, popt, dose_scale=1.0):
        popt = np.array(popt, dtype=float)
        popt[2] = np.exp(popt[2]) * dose_scale
        return popt

    @classmethod
    def undo_dose_scale(cls, popt, dose_scale):
        """Undo dose scaling on linear-space popt (EC50 is at index 2)."""
        popt = np.array(popt, dtype=float)
        popt[2] *= dose_scale
        return popt

    @property
    def ec50(self):
        return self.popt[2]

    @property
    def e0(self):
        return 1.0

    @property
    def popt_rel(self):
        if self._popt_rel is None:
            self._popt_rel = self.popt.copy()
            self._popt_rel[1] /= self.divisor
        return self._popt_rel


class HillCurveLL2(HillCurveLL3u):
    fit_bounds = ((0.0, -np.inf), (np.inf, np.inf))
    # LL2 has no bounds at all in linear space; log-EC50 space is also unbounded
    fit_bounds_log = (-np.inf, np.inf)
    curve_fit_kwargs_log = {}
    # Fully unbounded fit uses LM solver, which requires an integer maxfev
    max_fit_evals = 0

    @classmethod
    def fit_fn(cls, x, b, e):
        """
        Two parameter log-logistic function ("Hill curve")

        Parameters
        ----------
        x: np.ndarray
            One-dimensional array of "x" values
        b: float
            Hill slope
        e: float
            EC50 value

        Returns
        -------
        np.ndarray
            Array of "y" values using the supplied curve fit parameters on "x"
        """
        return super(HillCurveLL3u, cls).fit_fn(x, b, 0.0, 1.0, e)

    @classmethod
    def fit_fn_log(cls, x, b, log_e):
        """LL2 Hill curve with log-transformed EC50."""
        return 1.0 / (1.0 + np.exp(b * (np.log(x) - log_e)))

    @classmethod
    def jac_fn_log(cls, x, b, log_e):
        """Analytic Jacobian of LL2 w.r.t. (b, log_e)."""
        log_x = np.log(x)
        s = np.exp(b * (log_x - log_e))
        denom = 1.0 + s
        denom_sq = denom * denom

        j_b = -s * (log_x - log_e) / denom_sq
        j_log_e = b * s / denom_sq

        jac = np.column_stack((j_b, j_log_e))
        jac[np.isnan(jac)] = 0.0
        return jac

    @classmethod
    def initial_guess(cls, x, y):
        b, _, _, e = super(HillCurveLL3u, cls).initial_guess(x, y)
        return b, e

    @classmethod
    def initial_guess_log(cls, x, y, dose_scale=1.0):
        b, e = cls.initial_guess(x, y)
        with np.errstate(divide='ignore'):
            log_e = np.log(e / dose_scale)
        return b, log_e

    @classmethod
    def transform_popt_from_log(cls, popt, dose_scale=1.0):
        popt = np.array(popt, dtype=float)
        popt[1] = np.exp(popt[1]) * dose_scale
        return popt

    @classmethod
    def undo_dose_scale(cls, popt, dose_scale):
        """Undo dose scaling on linear-space popt (EC50 is at index 1)."""
        popt = np.array(popt, dtype=float)
        popt[1] *= dose_scale
        return popt

    @property
    def ec50(self):
        return self.popt[1]

    @property
    def e0(self):
        return 1.0

    @property
    def emax(self):
        return 0.0

    @property
    def popt_rel(self):
        if self._popt_rel is None:
            self._popt_rel = self.popt.copy()
        return self._popt_rel


def _sanitise_initial_guess(p0, fit_bounds, scaled_doses, use_log_path, fit_cls):
    """Clip initial guess parameters to lie within fit bounds and sensible
    ranges.

    Prevents two common failure modes:
    1. Negative Hill slope (b) for bounded models (LL3u/LL2) raising
       ``ValueError: Initial guess is outside provided bounds``.
    2. Wildly extrapolated EC50 (log_e far outside dose range) causing
       Levenberg-Marquardt maxfev exhaustion.
    """
    p0 = list(p0)

    # --- Clip each element to within its bounds ---
    lb, ub = fit_bounds
    if np.ndim(lb) == 0:
        lb = np.full(len(p0), lb)
    if np.ndim(ub) == 0:
        ub = np.full(len(p0), ub)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    for i in range(len(p0)):
        if np.isnan(p0[i]) or np.isinf(p0[i]):
            p0[i] = 0.5 * (lb[i] + ub[i]) if np.isfinite(lb[i] + ub[i]) else 0.0
        if np.isfinite(lb[i]) and p0[i] < lb[i]:
            p0[i] = lb[i]
        if np.isfinite(ub[i]) and p0[i] > ub[i]:
            p0[i] = ub[i]

    # --- Fix degenerate Hill slope ---
    # b = 0 makes the gradient w.r.t. EC50 vanish → solver stalls.
    # For bounded models (b ≥ 0 lower bound), use 1.0 as a safe default.
    b_idx = 0  # Hill slope is always the first parameter
    if np.isfinite(lb[b_idx]) and lb[b_idx] >= 0.0 and p0[b_idx] < 0.01:
        p0[b_idx] = 1.0

    # --- Clamp EC50 / log_e to ±1 decade beyond measured dose range ---
    positive_doses = scaled_doses[scaled_doses > 0]
    if len(positive_doses) > 0:
        log_min = np.log(np.min(positive_doses))
        log_max = np.log(np.max(positive_doses))
        margin = np.log(10.0)  # 1 decade
        log_lo = log_min - margin
        log_hi = log_max + margin

        if use_log_path:
            # log_e is the last parameter
            e_idx = len(p0) - 1
            if p0[e_idx] < log_lo:
                p0[e_idx] = log_lo
            elif p0[e_idx] > log_hi:
                p0[e_idx] = log_hi
        else:
            # EC50 in linear space is the last parameter
            e_idx = len(p0) - 1
            e_lo = np.exp(log_lo)
            e_hi = np.exp(log_hi)
            if p0[e_idx] < e_lo:
                p0[e_idx] = e_lo
            elif p0[e_idx] > e_hi:
                p0[e_idx] = e_hi

    return p0


def fit_drc(
    doses,
    responses,
    response_std_errs=None,
    fit_cls=HillCurveLL4,
    null_rejection_threshold=0.05,
    ctrl_dose_test=False,
):
    """
    Fit a dose response curve

    Parameters
    ----------
    doses: np.ndarray
        Array of dose values
    responses: np.ndarray
        Array of response values, e.g. viability, DIP rates
    response_std_errs: np.ndarray, optional
        Array of fit standard errors for the response values
    fit_cls: Class
        Class to use for fitting (default: 4 parameter log logistic
        "Hill" curve)
    null_rejection_threshold: float, optional
        p-value for rejecting curve fit against no effect "flat" response
        model by F-test (default: 0.05). Set to None to skip test.
    ctrl_dose_test: boolean
        If True, the minimum dose is assumed to represent control values (in
        DIP rate curves), and will reject fits where E0 is greater than a
        standard deviation higher than the mean of the control response values.
        Leave as False to skip the test.

    Returns
    -------
    HillCurve
        A HillCurve object containing the fit parameters
    """
    # Remove any NaNs
    response_nans = np.isnan(responses)
    if np.any(response_nans):
        doses = doses[~response_nans]
        responses = responses[~response_nans]
        if response_std_errs is not None:
            response_std_errs = response_std_errs[~response_nans]

    # Sort the data - curve fit can be sensitive to order!
    try:
        if response_std_errs is None:
            doses, responses = zip(*sorted(zip(doses, responses)))
        else:
            doses, responses, response_std_errs = zip(
                *sorted(zip(doses, responses, response_std_errs))
            )
    except ValueError:
        # Occurs when doses/responses is empty
        return None

    doses = np.asarray(doses, dtype=float)
    responses = np.asarray(responses, dtype=float)

    # Decide whether to use the log-EC50 parameterisation.
    #
    # The log-EC50 transform has two benefits:
    #   1. Removes the positivity bound on EC50, enabling the faster
    #      Levenberg-Marquardt solver when all other bounds are also absent.
    #   2. Provides an analytic Jacobian (avoids ~4 finite-difference evals
    #      per optimiser step).
    #
    # Path selection per curve class:
    #
    #   LL4 (fully unbounded): log path + Python fn + no Jacobian.
    #     Log-transform removes all bounds → scipy uses LM (unconstrained),
    #     which is faster than TRF on this problem.
    #
    #   LL3u / LL2 (bounded b and/or c): log path + Python fn + Python Jacobian.
    #     These classes still have bounds, so TRF is always used.  The analytic
    #     Jacobian reduces TRF iterations and dominates any other optimisation.
    #
    #   Fallback (no log variant defined): linear path, plain Python fn.
    has_log_variant = hasattr(fit_cls, 'fit_fn_log')
    use_log_path = has_log_variant

    if use_log_path:
        fit_fn = fit_cls.fit_fn_log
        # LL4 uses LM (unconstrained); Jacobian not needed and adds overhead
        jac_fn = None if fit_cls is HillCurveLL4 else fit_cls.jac_fn_log
        fit_bounds = fit_cls.fit_bounds_log
        curve_fit_kwargs = fit_cls.curve_fit_kwargs_log
    else:
        fit_fn = fit_cls.fit_fn
        jac_fn = None
        fit_bounds = fit_cls.fit_bounds
        curve_fit_kwargs = fit_cls.curve_fit_kwargs

    # Dose-scale normalisation: centre doses on a log scale to reduce
    # floating-point precision issues across extreme concentration ranges.
    # Mirrors the approach in the synergy package.  Only applied on the log
    # path where the required back-transform methods are available.
    can_dose_scale = use_log_path and hasattr(fit_cls, 'transform_popt_from_log')
    if can_dose_scale:
        positive_doses = doses[doses > 0]
        dose_scale = (
            np.exp(np.median(np.log(positive_doses)))
            if len(positive_doses) > 0
            else 1.0
        )
    else:
        dose_scale = 1.0
    scaled_doses = doses / dose_scale

    if use_log_path:
        curve_initial_guess = fit_cls.initial_guess_log(
            scaled_doses, responses, dose_scale=1.0
        )
    else:
        curve_initial_guess = fit_cls.initial_guess(scaled_doses, responses)

    curve_initial_guess = _sanitise_initial_guess(
        curve_initial_guess, fit_bounds, scaled_doses, use_log_path, fit_cls
    )

    param_count = len(curve_initial_guess)
    if len(scaled_doses) < param_count:
        return None

    popt = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                popt, pcov = scipy.optimize.curve_fit(
                    fit_fn,
                    scaled_doses,
                    responses,
                    bounds=fit_bounds,
                    p0=curve_initial_guess,
                    sigma=response_std_errs,
                    maxfev=fit_cls.max_fit_evals,
                    jac=jac_fn,
                    **curve_fit_kwargs,
                )
    except RuntimeError:
        pass  # fall through to fallback or return None below
    except ValueError:
        pass  # fall through to fallback (e.g. degenerate data)
    except TypeError as te:
        # This occurs if there are fewer data points than parameters
        te_str = str(te)
        if 'Improper input:' in te_str or te_str.startswith(
            'The number of func parameters'
        ):
            return None
        else:
            raise

    # If log-EC50 fit did not converge, retry with the original linear-EC50
    # parameterisation (no Jacobian, but handles degenerate cases better).
    if popt is None and use_log_path:
        fallback_guess = fit_cls.initial_guess(scaled_doses, responses)
        fallback_guess = _sanitise_initial_guess(
            fallback_guess,
            fit_cls.fit_bounds,
            scaled_doses,
            use_log_path=False,
            fit_cls=fit_cls,
        )
        try:
            if len(scaled_doses) < len(fallback_guess):
                return None

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    popt, pcov = scipy.optimize.curve_fit(
                        fit_cls.fit_fn,
                        scaled_doses,
                        responses,
                        bounds=fit_cls.fit_bounds,
                        p0=fallback_guess,
                        sigma=response_std_errs,
                        maxfev=fit_cls.max_fit_evals,
                        **fit_cls.curve_fit_kwargs,
                    )
        except RuntimeError:
            return None
        except ValueError:
            return None
        except TypeError as te:
            te_str = str(te)
            if 'Improper input:' in te_str or te_str.startswith(
                'The number of func parameters'
            ):
                return None
            else:
                raise
        # Fallback popt is in linear EC50 + scaled-dose space; undo dose scale
        if hasattr(fit_cls, 'undo_dose_scale') and dose_scale != 1.0:
            popt = fit_cls.undo_dose_scale(popt, dose_scale)
        use_log_path = False  # popt is already in linear space

    if popt is None or any(np.isnan(popt)):
        return None

    # Back-transform from log-EC50 space and undo dose scaling
    if use_log_path:
        popt = fit_cls.transform_popt_from_log(popt, dose_scale=dose_scale)
    elif dose_scale != 1.0 and hasattr(fit_cls, 'undo_dose_scale'):
        # Linear-space primary fit: undo dose scaling
        popt = fit_cls.undo_dose_scale(popt, dose_scale)

    fit_obj = fit_cls(popt)

    if null_rejection_threshold is not None:
        null_response_value = fit_cls.null_response_fn(responses)
        response_curve = fit_obj.fit(doses)

        # F test vs flat linear "no effect" fit
        ssq_model = ((response_curve - responses) ** 2).sum()
        ssq_null = ((null_response_value - responses) ** 2).sum()

        df = len(doses) - len(popt)

        with np.errstate(divide='ignore', invalid='ignore'):
            f_ratio = (ssq_null - ssq_model) / (ssq_model / df)
        p = 1 - scipy.stats.f.cdf(f_ratio, 1, df)

        if p > null_rejection_threshold:
            return HillCurveNull(null_response_value)

    if fit_obj.ec50 < np.min(doses):
        # Reject fit if EC50 less than min dose
        return None

    if fit_cls is not HillCurveLL4 and fit_obj.ec50 > 10.0 * np.max(doses):
        # Reject fit if EC50 more than a decade above max dose — the curve's
        # transition lies well beyond the measured range.  A 10x margin keeps
        # borderline fits where the drug effect is partially observed.
        # Only for constrained models (LL3u/LL2); LL4 fits with EC50 above
        # the dose range can still characterise the observed partial response.
        return None

    if ctrl_dose_test:
        responses = np.array(responses)
        controls = responses[np.equal(doses, np.min(doses))]
        if len(controls) >= E0_TEST_MIN_CTRLS_STD_DEV:
            ctrl_resp_thresh = np.mean(controls) + np.std(controls)
        else:
            ctrl_resp_thresh = E0_TEST_MULTIPLIER * np.mean(controls)
        if fit_obj.e0 > ctrl_resp_thresh:
            return None

    return fit_obj


def _response_transform(y, c_val, d_val):
    return np.log((d_val - y) / (y - c_val))


def _find_be_ll4(
    x,
    y,
    c_val,
    d_val,
    slope_scaling_factor=1,
    dose_transform=np.log,
    dose_inv_transform=np.exp,
):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        dose_transform(x), _response_transform(y, c_val, d_val)
    )
    b_val = slope_scaling_factor * slope
    e_val = dose_inv_transform(-intercept / (slope_scaling_factor * b_val))

    return b_val, e_val


def _find_cd_ll4(y, scale=0.001):
    ymin = np.min(y)
    ymax = np.max(y)
    len_y_range = scale * (ymax - ymin)

    return ymin - len_y_range, ymax + len_y_range


def _get_control_responses(ctrl_dip_data, dataset, cl_name, dip_grp):
    if ctrl_dip_data is None:
        return None

    if dataset is not None and 'dataset' in ctrl_dip_data.index.names:
        ctrl_dip_data_cl = ctrl_dip_data.loc[dataset]
    else:
        ctrl_dip_data_cl = ctrl_dip_data

    ctrl_dip_data_cl = ctrl_dip_data_cl.loc[cl_name]

    # Only use controls from the same plates as the expt
    if 'plate' in dip_grp.index.names:
        plates = dip_grp.index.get_level_values('plate').unique().values
    elif 'plate' in dip_grp.columns:
        plates = dip_grp['plate'].unique()
    else:
        plates = []

    ctrl_dip_data_cl = ctrl_dip_data_cl.loc[
        ctrl_dip_data_cl.index.isin(plates, level='plate')
    ]

    if ctrl_dip_data_cl.empty:
        return None

    return ctrl_dip_data_cl


# Parameter fitting section #
def aa_obs(responses, doses=None):
    """
    Activity Area (observed)

    Parameters
    ----------
    responses: np.array or pd.Series
        Response values, with dose values in the Index if a Series is
        supplied
    doses: np.array or None
        Dose values - only required if responses is not a pd.Series

    Returns
    -------
    float
        Activity area (observed)
    """
    if doses is None:
        responses = responses.groupby('dose').agg('mean')
        doses = responses.index.get_level_values('dose')
    responses_shifted = 1.0 - np.minimum(responses, 1.0)
    # Ensure array is sorted by dose
    positions = np.argsort(doses)
    responses_shifted = responses_shifted[positions]
    doses = doses[positions]
    return np.trapezoid(responses_shifted, np.log10(doses))


def fit_params_minimal(
    ctrl_data,
    expt_data,
    fit_cls=HillCurveLL4,
    ctrl_dose_fn=lambda doses: np.min(doses) / CTRL_DOSE_DIVISOR,
):
    """
    Fit dose response curves to DIP or viability, and calculate statistics

    This function only fits curves and stores basic fit parameters. Use
    :func:`fit_params` for more statistics and parameters.

    Parameters
    ----------
    ctrl_data: pd.DataFrame or None
        Control DIP rates from :func:`dip_rates` or :func:`ctrl_dip_rates`.
        Set to None to not use control data.
    expt_data: pd.DataFrame
        Experiment (non-control) DIP rates from :func:`dip_rates` or
        :func:`expt_dip_rates`
    fit_cls: Class
        Class to use for curve fitting (default: :func:`HillCurveLL4`)
    ctrl_dose_fn: function
        Function to use to set an effective "dose" (non-zero) for controls.
        Takes the list of experiment doses as an argument.

    Returns
    -------
    pd.DataFrame
        DataFrame containing DIP rate curve fits and parameters
    """
    expt_data_orig = expt_data

    if 'dataset' in expt_data.index.names:
        datasets = expt_data.index.get_level_values('dataset').unique()
    else:
        datasets = None
    cell_lines = expt_data.index.get_level_values('cell_line').unique()
    drugs = expt_data.index.get_level_values('drug').unique()

    has_drug_combos = drugs.map(len).max() > 1
    if has_drug_combos:
        combo_mask = expt_data.index.get_level_values('drug').map(len) > 1
        n_combo = int(combo_mask.sum())
        warnings.warn(
            f'{n_combo} combination well(s) skipped; drug combination curve '
            f'fitting is not yet implemented.',
            DrugCombosWarning,
            stacklevel=2,
        )
        expt_data = expt_data[~combo_mask]
        if expt_data.empty:
            raise ValueError(
                'No single-drug experiment wells remain after skipping combination wells'
            )
        drugs = expt_data.index.get_level_values('drug').unique()

    # Unwrap single-drug tuples
    expt_data = expt_data.reset_index(['drug', 'dose'])
    expt_data['drug'] = expt_data['drug'].apply(lambda x: x[0])
    expt_data['dose'] = expt_data['dose'].apply(lambda x: x[0])
    expt_data.set_index(['drug', 'dose'], append=True, inplace=True)
    drugs = expt_data.index.get_level_values('drug').unique()

    if len(drugs) > 1 and len(cell_lines) == 1:
        group_by = ['drug']
    elif len(cell_lines) > 1 and len(drugs) == 1:
        group_by = ['cell_line']
    else:
        group_by = ['cell_line', 'drug']

    if datasets is not None and len(datasets) > 1:
        if len(cell_lines) == 1 and len(drugs) == 1:
            group_by = ['dataset']
        else:
            group_by = ['dataset'] + group_by

    fit_params = []

    is_viability = 'viability' in expt_data.columns

    for group_name, dip_grp in expt_data.groupby(
        level=group_by[0] if len(group_by) == 1 else group_by
    ):
        if 'dataset' in group_by:
            if len(group_by) > 1:
                dataset = group_name[group_by.index('dataset')]
            else:
                dataset = group_name
        elif datasets is not None:
            dataset = datasets[0]
        else:
            dataset = None

        if 'cell_line' in group_by:
            if len(group_by) > 1:
                cl_name = group_name[group_by.index('cell_line')]
            else:
                cl_name = group_name
        else:
            cl_name = cell_lines[0]

        if 'drug' in group_by:
            if len(group_by) > 1:
                dr_name = group_name[group_by.index('drug')]
            else:
                dr_name = group_name
        else:
            dr_name = drugs[0]

        doses_expt = dip_grp.index.get_level_values('dose').to_numpy()

        if is_viability:
            resp_expt = dip_grp['viability'].to_numpy()
            doses = doses_expt
            fit_obj = fit_drc(
                doses_expt,
                resp_expt,
                response_std_errs=None,
                null_rejection_threshold=None,
                fit_cls=fit_cls,
            )
            aa_obs_val = aa_obs(resp_expt, doses_expt)
        else:
            if (
                dataset is None
                and ctrl_data is not None
                and 'dataset' in ctrl_data.index.names
            ):
                raise ValueError(
                    'Experimental data does not have "dataset" '
                    'in index, but control data does. Please '
                    'make sure "dataset" is in both dataframes, '
                    'or neither.'
                )

            ctrl_dip_data_cl = _get_control_responses(
                ctrl_data, dataset, cl_name, dip_grp
            )
            dip_ctrl = []
            dip_ctrl_std_err = []
            if ctrl_dip_data_cl is not None:
                dip_ctrl = ctrl_dip_data_cl['dip_rate'].values
                dip_ctrl_std_err = ctrl_dip_data_cl['dip_fit_std_err'].values

            n_controls = len(dip_ctrl)
            ctrl_dose_val = ctrl_dose_fn(doses_expt)
            doses_ctrl = np.repeat(ctrl_dose_val, n_controls)
            doses = np.concatenate((doses_ctrl, doses_expt))
            resp_expt = dip_grp['dip_rate'].values
            dip_all = np.concatenate((dip_ctrl, resp_expt))
            dip_std_errs = np.concatenate(
                (dip_ctrl_std_err, dip_grp['dip_fit_std_err'].values)
            )

            try:
                fit_obj = fit_drc(
                    doses, dip_all, dip_std_errs, fit_cls=fit_cls, ctrl_dose_test=True
                )
            except KeyError:
                fit_obj = None

            if fit_obj is not None:
                aa_obs_val = aa_obs(resp_expt / fit_obj.divisor, doses_expt)
            elif len(dip_ctrl) > 0:
                # If no fit, use average ctrl response to put DIP values
                # on relative scale
                aa_obs_val = aa_obs(resp_expt / np.mean(dip_ctrl), doses_expt)
            else:
                aa_obs_val = None

        max_dose_measured = np.max(doses)
        min_dose_measured = np.min(doses)

        fit_data = dict(
            dataset_id=dataset if dataset is not None else '',
            cell_line=cl_name,
            drug=dr_name,
            fit_obj=fit_obj,
            min_dose_measured=min_dose_measured,
            max_dose_measured=max_dose_measured,
            emax_obs=np.min(resp_expt),
            aa_obs=aa_obs_val,
        )

        fit_params.append(fit_data)

    df_params = pd.DataFrame(fit_params)
    df_params.set_index(['dataset_id', 'cell_line', 'drug'], inplace=True)

    df_params.attrs['drmetric'] = 'viability' if is_viability else 'dip'
    if is_viability:
        df_params.attrs.update(expt_data_orig.attrs)

    return df_params


def _calc_e(row, ec_lbl, relative=False):
    if row.fit_obj is None:
        return None

    ec_val = row[ec_lbl]
    if ec_val is None:
        return None

    return row.fit_obj.fit_rel(ec_val) if relative else row.fit_obj.fit(ec_val)


def _attach_extra_params(
    base_params,
    custom_ic_concentrations=frozenset(),
    custom_ec_concentrations=frozenset(),
    custom_e_values=frozenset(),
    custom_e_rel_values=frozenset(),
    include_aa=False,
    include_auc=False,
    include_hill=False,
    include_emax=False,
    include_einf=False,
):
    datasets = base_params.index.get_level_values('dataset_id').unique()
    if len(datasets) == 1 and datasets[0] == '':
        datasets = None

    cell_lines = base_params.index.get_level_values('cell_line').unique()
    drugs = base_params.index.get_level_values('drug').unique()

    if len(drugs) > 1 and len(cell_lines) == 1:
        group_by = ['drug']
    elif len(cell_lines) > 1 and len(drugs) == 1:
        group_by = ['cell_line']
    else:
        group_by = ['cell_line', 'drug']

    if datasets is not None and len(datasets) > 1:
        if len(cell_lines) == 1 and len(drugs) == 1:
            group_by = ['dataset_id']
        else:
            group_by = ['dataset_id'] + group_by

    index_names = base_params.index.names

    def _generate_label(index):
        group_name_components = []

        if 'dataset_id' in group_by:
            dataset = index[index_names.index('dataset_id')]
            group_name_components.append(str(dataset))

        if 'cell_line' in group_by:
            cl_name = index[index_names.index('cell_line')]
            group_name_components.append(str(cl_name))

        if 'drug' in group_by:
            dr_name = index[index_names.index('drug')]
            group_name_components.append(str(dr_name))

        return '\n'.join(group_name_components)

    base_params['label'] = base_params.index.map(_generate_label)

    is_viability = base_params.attrs.get('drmetric') == 'viability'

    # Determine which per-row columns are needed
    need_ec50 = 50 in custom_ec_concentrations
    ic_nums = sorted(custom_ic_concentrations)
    custom_ec_concentrations = set(custom_ec_concentrations)
    custom_ec_concentrations.discard(50)
    custom_ec_concentrations = custom_ec_concentrations.union(custom_e_values)
    custom_ec_concentrations = custom_ec_concentrations.union(custom_e_rel_values)
    ec_nums = sorted(custom_ec_concentrations)
    e_nums = sorted(custom_e_values)
    e_rel_nums = sorted(custom_e_rel_values)

    need_any = (
        ic_nums
        or need_ec50
        or include_emax
        or include_einf
        or include_aa
        or include_auc
        or include_hill
        or ec_nums
    )

    if need_any:
        fit_objs = base_params['fit_obj'].to_numpy()
        max_doses = base_params['max_dose_measured'].to_numpy()
        min_doses = base_params['min_dose_measured'].to_numpy()
        n = len(fit_objs)

        # Pre-allocate output lists
        ic_cols = {num: [None] * n for num in ic_nums}
        ec50_col = [None] * n if need_ec50 else None
        emax_col = [None] * n if include_emax else None
        einf_col = [None] * n if include_einf else None
        aa_col = [None] * n if include_aa else None
        auc_col = [None] * n if include_auc else None
        hill_col = [None] * n if include_hill else None
        ec_cols = {num: [None] * n for num in ec_nums}

        for i in range(n):
            fo = fit_objs[i]
            if not fo:
                continue
            max_d = max_doses[i]
            min_d = min_doses[i]

            for ic_num in ic_nums:
                ic_v = fo.ic(ic_num=ic_num)
                if ic_v is not None:
                    ic_cols[ic_num][i] = float(min(max(ic_v, min_d), max_d))

            if need_ec50 and fo.ec50 is not None:
                ec50_col[i] = float(min(max(fo.ec50, min_d), max_d))

            if include_emax:
                emax_col[i] = fo.fit(max_d)

            if include_einf:
                einf_col[i] = fo.emax

            if include_aa:
                aa_col[i] = fo.aa(min_conc=min_d, max_conc=max_d)

            if include_auc:
                auc_col[i] = fo.auc(min_conc=min_d)

            if include_hill:
                hill_col[i] = fo.hill_slope

            for ec_num in ec_nums:
                ec_v = fo.ec(ec_num=ec_num)
                if ec_v is not None:
                    ec_cols[ec_num][i] = float(min(max(ec_v, min_d), max_d))

        for ic_num in ic_nums:
            base_params['ic{:d}'.format(ic_num)] = ic_cols[ic_num]

        if need_ec50:
            base_params['ec50'] = ec50_col

        if include_emax:
            base_params['emax'] = emax_col

        if include_einf:
            base_params['einf'] = einf_col

        if include_aa:
            base_params['aa'] = aa_col

        if include_auc:
            base_params['auc'] = auc_col

        if include_hill:
            base_params['hill'] = hill_col

        for ec_num in ec_nums:
            base_params['ec{:d}'.format(ec_num)] = ec_cols[ec_num]

    if not is_viability and include_emax:
        divisor = base_params['fit_obj'].apply(lambda fo: fo.divisor if fo else None)
        base_params['emax_rel'] = base_params['emax'] / divisor
        base_params['emax_obs_rel'] = base_params['emax_obs'] / divisor

    for e_num in e_nums:
        base_params['e{:d}'.format(e_num)] = base_params.apply(
            _calc_e, args=('ec{:d}'.format(e_num), False), axis=1
        )

    for e_num in e_rel_nums:
        base_params['e{:d}_rel'.format(e_num)] = base_params.apply(
            _calc_e, args=('ec{:d}'.format(e_num), True), axis=1
        )

    return base_params


def _attach_response_values(df_params, ctrl_dip_data, expt_dip_data, ctrl_dose_fn):
    is_viability = df_params.attrs.get('drmetric') == 'viability'
    data_list = []
    if 'dataset' not in expt_dip_data.index.names:
        expt_dip_data = expt_dip_data.copy()
        expt_dip_data['dataset'] = ''
        old_index_cols = expt_dip_data.index.names
        expt_dip_data.reset_index(inplace=True)
        expt_dip_data.set_index(['dataset'] + old_index_cols, inplace=True)
    for grp, dip_grp in expt_dip_data.groupby(
        ['dataset', 'cell_line', 'drug'], sort=False
    ):
        # Assumes drug combinations have been ruled out by fit_params_minimal
        doses_expt = [d[0] for d in dip_grp.index.get_level_values('dose').values]
        fit_data = {'dataset_id': grp[0], 'cell_line': grp[1], 'drug': grp[2][0]}

        ctrl_dip_data_cl = _get_control_responses(
            ctrl_dip_data, grp[0], grp[1], dip_grp
        )
        if ctrl_dip_data_cl is not None:
            if is_viability:
                ctrl_dip_data_cl = ctrl_dip_data_cl.to_frame()
            n_controls = len(ctrl_dip_data_cl.index)
            ctrl_dose_val = ctrl_dose_fn(doses_expt)
            doses_ctrl = np.repeat(ctrl_dose_val, n_controls)
            ctrl_dip_data_cl['dose'] = doses_ctrl
            ctrl_dip_data_cl.reset_index('well_id', inplace=True)
            ctrl_dip_data_cl.set_index(['dose', 'well_id'], inplace=True)
            if is_viability:
                fit_data['viability_ctrl'] = ctrl_dip_data_cl['value']
            else:
                fit_data['dip_ctrl'] = ctrl_dip_data_cl['dip_rate']

        if is_viability:
            fit_data['viability_time'] = dip_grp['timepoint'].values
            fit_data['viability'] = pd.Series(
                data=dip_grp['viability'].values,
                index=[doses_expt, dip_grp.index.get_level_values('well_id')],
            )
            fit_data['viability'].index.rename(['dose', 'well_id'], inplace=True)
        else:
            fit_data['dip_expt'] = pd.Series(
                data=dip_grp['dip_rate'].values,
                index=[doses_expt, dip_grp.index.get_level_values('well_id')],
            )
            fit_data['dip_expt'].index.rename(['dose', 'well_id'], inplace=True)

        data_list.append(fit_data)

    df = pd.DataFrame(data_list)
    df.set_index(['dataset_id', 'cell_line', 'drug'], inplace=True)
    df_params_old = df_params
    df_params = pd.concat([df, df_params], axis=1)

    df_params.attrs.update(df_params_old.attrs)

    return df_params


def fit_params(
    ctrl_data,
    expt_data,
    fit_cls=HillCurveLL4,
    ctrl_dose_fn=lambda doses: np.min(doses) / CTRL_DOSE_DIVISOR,
):
    """
    Fit dose response curves to DIP rates or viability data

    This method computes parameters including IC50, EC50, AUC, AA,
    Hill coefficient, and Emax. For a faster version,
    see :func:`fit_params_minimal`.

    Parameters
    ----------
    ctrl_data: pd.DataFrame or None
        Control DIP rates from :func:`dip_rates` or :func:`ctrl_dip_rates`.
        Set to None to not use control data.
    expt_data: pd.DataFrame
        Experiment (non-control) DIP rates from :func:`dip_rates` or
        :func:`expt_dip_rates`, or viability data from :func:`viability`
    fit_cls: Class
        Class to use for curve fitting (default: :func:`HillCurveLL4`)
    ctrl_dose_fn: function
        Function to use to set an effective "dose" (non-zero) for controls.
        Takes the list of experiment doses as an argument.

    Returns
    -------
    pd.DataFrame
        DataFrame containing DIP rate curve fits and parameters
    """
    base_params = fit_params_minimal(ctrl_data, expt_data, fit_cls, ctrl_dose_fn)

    return fit_params_from_base(
        base_params,
        ctrl_data,
        expt_data,
        ctrl_dose_fn=ctrl_dose_fn,
        custom_ic_concentrations={50},
        custom_ec_concentrations={50},
        include_auc=True,
        include_aa=True,
        include_hill=True,
        include_emax=True,
        include_einf=True,
        include_response_values=True,
    )


def fit_params_from_base(
    base_params,
    ctrl_data=None,
    expt_data=None,
    ctrl_dose_fn=lambda doses: np.min(doses) / CTRL_DOSE_DIVISOR,
    custom_ic_concentrations=frozenset(),
    custom_ec_concentrations=frozenset(),
    custom_e_values=frozenset(),
    custom_e_rel_values=frozenset(),
    include_aa=False,
    include_auc=False,
    include_hill=False,
    include_emax=False,
    include_einf=False,
    include_response_values=True,
):
    """
    Attach additional parameters to basic set of fit parameters
    """
    df_params = _attach_extra_params(
        base_params,
        custom_ic_concentrations,
        custom_ec_concentrations,
        custom_e_values,
        custom_e_rel_values,
        include_aa,
        include_auc,
        include_hill,
        include_emax,
        include_einf,
    )

    if include_response_values:
        df_params = _attach_response_values(
            df_params, ctrl_data, expt_data, ctrl_dose_fn
        )

    return df_params


def is_param_truncated(df_params, param_name):
    """
    Checks if parameter values are truncated at boundaries of measured range

    Parameters
    ----------
    df_params: pd.DataFrame
        DataFrame of DIP curve fits with parameters from :func:`fit_params`
    param_name: str
        Name of a parameter, e.g. 'ic50'

    Returns
    -------
    np.ndarray
        Array of booleans showing whether each entry in the DataFrame is
        truncated
    """
    values = df_params[param_name].fillna(value=np.nan)
    return np.isclose(
        values,
        df_params['max_dose_measured'],
        atol=PARAM_EQUAL_ATOL,
        rtol=PARAM_EQUAL_RTOL,
    ) | np.isclose(
        values,
        df_params['min_dose_measured'],
        atol=PARAM_EQUAL_ATOL,
        rtol=PARAM_EQUAL_RTOL,
    )
