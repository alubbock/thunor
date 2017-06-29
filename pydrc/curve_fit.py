import numpy as np
import scipy.optimize
import scipy.stats


def ll4(x, b, c, d, e):
    """
    Four parameter log-logistic function ("Hill curve")

     - b: Hill slope
     - c: min response
     - d: max response
     - e: EC50
     """
    return c+(d-c)/(1+np.exp(b*(np.log(x)-np.log(e))))


# Fitting function
def fit_drc(doses, dip_rates, dip_std_errs=None, hill_fn=ll4):
    dip_rate_nans = np.isnan(dip_rates)
    if np.any(dip_rate_nans):
        doses = doses[~dip_rate_nans]
        dip_rates = dip_rates[~dip_rate_nans]
        if dip_std_errs is not None:
            dip_std_errs = dip_std_errs[~dip_rate_nans]
    popt = None
    curve_initial_guess = ll4_initials(doses, dip_rates)
    try:
        popt, pcov = scipy.optimize.curve_fit(hill_fn,
                                              doses,
                                              dip_rates,
                                              p0=curve_initial_guess,
                                              sigma=dip_std_errs
                                              )

        if popt[1] > popt[2] or popt[0] < 0:
            # TODO: Maybe try another fit of some kind?
            popt = None
    except RuntimeError:
        pass

    popt_rel = None
    if popt is None:
        divisor = np.mean(dip_rates)
    else:
        divisor = popt[2]
        if divisor > 0:
            # Are cells growing in control?
            popt_rel = popt.copy()
            popt_rel[1] /= divisor
            popt_rel[2] = 1

    return popt, popt_rel, divisor


# Functions for finding initial parameter estimates for curve fitting
def ll4_initials(x, y):
    c_val, d_val = _find_cd_ll4(y)
    b_val, e_val = _find_be_ll4(x, y, c_val, d_val)

    return b_val, c_val, d_val, e_val


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
