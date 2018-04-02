import numpy as np
import scipy.optimize
import scipy.stats
from abc import abstractclassmethod, abstractmethod


class HillCurve(object):
    """ Base class defining Hill/log-logistic curve functionality """
    def __init__(self, popt):
        self.popt = popt

    @abstractclassmethod
    def fit_fn(cls, x, *params):
        pass

    def fit(self, x):
        return self.fit_fn(x, *self.popt)

    @abstractclassmethod
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

    def hill_slope(self):
        return None


class HillCurveLL4(HillCurve):
    def __init__(self, popt):
        super(HillCurveLL4, self).__init__(popt)
        self._popt_rel = None

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
            Minimum response (lower plateau)
        d: float
            Maximum response (upper plateau)
        e: float
            EC50 value

        Returns
        -------
        np.ndarray
            Array of "y" values using the supplied curve fit parameters on "x"
        """
        return c + (d - c) / (1 + np.exp(b * (np.log(x) - np.log(e))))

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

        icN = self.ec50 * (ic_frac / (1 - ic_frac - (emax / e0))) ** (
                           1 / self.hill_slope)

        if np.isnan(icN):
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
        float
            Area under the curve (AUC) value
        """
        emax = self.emax
        if not isinstance(emax, float):
            return None
        e0 = self.e0
        if emax > e0:
            emax, e0 = e0, emax
        min_conc_hill = min_conc ** self.hill_slope
        return (np.log10(
            (self.ec50 ** self.hill_slope + min_conc_hill) / min_conc_hill) /
                self.hill_slope) * ((e0 - emax) / e0)

    def aa(self, max_conc, emax_obs=None):
        """
        Find the activity area (area over the curve)

        Parameters
        ----------
        max_conc: float
            Maximum concentration to consider for fitting the curve
        emax_obs: float, optional
            Observed Emax value

        Returns
        -------
        float
            Activity area value
        """
        emax = self.emax
        if not isinstance(emax, float):
            return None
        e0 = self.e0
        if emax > e0:
            emax, e0 = e0, emax

        if emax_obs is not None:
            emax = emax_obs

        ec50_hill = self.ec50 ** self.hill_slope

        return np.log10((ec50_hill + max_conc ** self.hill_slope) /
                        ec50_hill) * ((e0 - emax) / e0) / self.hill_slope

    @property
    def divisor(self):
        return max(self.emax, self.e0)

    @property
    def popt_rel(self):
        if self._popt_rel is None:
            self._popt_rel = self.popt.copy()
            self.popt_rel[2] /= self.divisor
            self.popt_rel[1] /= self.divisor
        return self._popt_rel

    def fit_rel(self, x):
        return self.fit_fn(x, *self.popt_rel)


class HillCurveLL3u(HillCurveLL4):
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
            Minimum response (lower plateau)
        e: float
            EC50 value

        Returns
        -------
        np.ndarray
            Array of "y" values using the supplied curve fit parameters on "x"
        """
        return super(HillCurveLL3u, cls).fit_fn(x, b, c, 1.0, e)

    @classmethod
    def initial_guess(cls, x, y):
        b, c, _, e = super(HillCurveLL3u, cls).initial_guess(x, y)
        return b, c, e

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


def fit_drc(doses, responses, response_std_errs=None, fit_cls=HillCurveLL4,
            null_rejection_threshold=0.05, ctrl_dose=None):
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
    ctrl_dose: float, optional
        Enter the dose used to represent control values, if you wish to reject
        fits where E0 is not greater than a standard deviation higher than the
        mean of the control response values. Leave as None to skip the test.

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
    if response_std_errs is None:
        doses, responses = zip(*sorted(zip(doses, responses)))
    else:
        doses, responses, response_std_errs = zip(*sorted(zip(
            doses, responses, response_std_errs)))

    curve_initial_guess = fit_cls.initial_guess(doses, responses)
    try:
        popt, pcov = scipy.optimize.curve_fit(fit_cls.fit_fn,
                                              doses,
                                              responses,
                                              p0=curve_initial_guess,
                                              sigma=response_std_errs
                                              )
    except RuntimeError:
        # Some numerical issue with curve fitting
        return None

    if any(np.isnan(popt)):
        # Ditto
        return None

    fit_obj = fit_cls(popt)

    if null_rejection_threshold is not None:
        response_curve = fit_obj.fit(doses)

        # F test vs flat linear "no effect" fit
        ssq_model = ((response_curve - responses) ** 2).sum()
        ssq_null = ((np.mean(responses) - responses) ** 2).sum()

        df = len(doses) - 4

        f_ratio = (ssq_null-ssq_model)/(ssq_model/df)
        p = 1 - scipy.stats.f.cdf(f_ratio, 1, df)

        if p > null_rejection_threshold:
            return HillCurveNull(np.mean(responses))

    if fit_obj.ec50 < np.min(doses):
        # Reject fit if EC50 less than min dose
        return None

    if ctrl_dose is not None:
        controls = responses[np.equal(doses, ctrl_dose)]
        if fit_obj.e0 > (np.mean(controls) + np.std(controls)):
            return None

    return fit_obj


def _response_transform(y, c_val, d_val):
    return np.log((d_val - y) / (y - c_val))


def _find_be_ll4(x, y, c_val, d_val, slope_scaling_factor=1,
                 dose_transform=np.log,
                 dose_inv_transform=np.exp):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        dose_transform(x),
        _response_transform(y, c_val, d_val)
    )
    b_val = slope_scaling_factor * slope
    e_val = dose_inv_transform(-intercept / (slope_scaling_factor * b_val))

    return b_val, e_val


def _find_cd_ll4(y, scale=0.001):
    ymin = np.min(y)
    ymax = np.max(y)
    len_y_range = scale * (ymax - ymin)

    return ymin - len_y_range, ymax + len_y_range
