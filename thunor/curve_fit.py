import numpy as np
import scipy.optimize
import scipy.stats


def ll4(x, b, c, d, e):
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
    return c+(d-c)/(1+np.exp(b*(np.log(x)-np.log(e))))


def ll4_initials(x, y):
    """
    Heuristic function for initial fit values for ll4 function

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


def fit_drc(doses, responses, response_std_errs=None, hill_fn=ll4,
            curve_initial_guess_fn=ll4_initials,
            null_rejection_threshold=0.05):
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
    hill_fn: function
        Function to use for fitting (default: 4 parameter log logistic
        "Hill" curve)
    curve_initial_guess_fn: function
        Function to use for initial parameter guess before curve fitting
    null_rejection_threshold: float
        p-value for rejecting curve fit against no effect "flat" response
        model by F-test (default: 0.05)

    Returns
    -------
    list
        The return value is a list with three entries:

        - tuple of (absolute scale) fit parameters
        - tuple of fit parameters on relative scale (max response=1)
        - float - the divisor used to convert the absolute fit parameters to
          relative scale

    """
    response_nans = np.isnan(responses)
    if np.any(response_nans):
        doses = doses[~response_nans]
        responses = responses[~response_nans]
        if response_std_errs is not None:
            response_std_errs = response_std_errs[~response_nans]
    curve_initial_guess = curve_initial_guess_fn(doses, responses)
    try:
        popt, pcov = scipy.optimize.curve_fit(hill_fn,
                                              doses,
                                              responses,
                                              p0=curve_initial_guess,
                                              sigma=response_std_errs
                                              )
    except RuntimeError:
        # Some numerical issue with curve fitting
        return None, None, None

    if any(np.isnan(popt)):
        # Ditto
        return None, None, None

    response_curve = hill_fn(doses, *popt)

    # F test vs flat linear "no effect" fit
    ssq_model = ((response_curve - responses) ** 2).sum()
    ssq_null = ((np.mean(responses) - responses) ** 2).sum()

    df = len(doses) - 4

    f_ratio = (ssq_null-ssq_model)/(ssq_model/df)
    p = 1 - scipy.stats.f.cdf(f_ratio, 1, df)

    if p > null_rejection_threshold:
        return None, None, np.mean(responses)

    if popt[3] < np.min(doses):
        # Reject fit if EC50 less than min dose
        return None, None, None

    divisor = max(popt[1], popt[2])
    popt_rel = popt.copy()
    popt_rel[2] /= divisor
    popt_rel[1] /= divisor

    return popt, popt_rel, divisor


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
