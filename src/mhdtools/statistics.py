import numpy as np
from scipy.optimize import curve_fit
import mhdtools


def mae(error_array):
    """Compute mean absolute error given an error field

    Parameters
    ----------
    error_array : numpy array
        Array of error values (data - reference, or reference - data)

    Returns
    -------
    float
        Mean absolute error of the data
    """
    return np.mean(abs(error_array))


def rmse(error_array):
    """Compute root-mean-squared error given an error field

    Parameters
    ----------
    error_array : numpy array
        Array of error values (data - reference, or reference - data)

    Returns
    -------
    float
        Root-mean-squared error of the data
    """
    return np.sqrt(np.mean(error_array**2))


def highPassFilter(array, threshold):
    """Given an array, set all entries with absolute value below a
    specified threshold to zero.

    Parameters
    ----------
    array : numpy array
        Array to be filtered
    threshold : float
        Absolute value below which all values should be set to zero

    Returns
    -------
    numpy array
        Array with high pass filter applied
    """
    filteredArray = array
    filteredArray[np.abs(array) < threshold] = 0
    return filteredArray


def fraction_rmse_comparison(data, reference_data):
    """Calculate the error of an array compared to an array of reference data, by
    finding the pointwise difference between both arrays, computing the
    root-mean-squared error of the data, and dividing the RMSE value by the mean of the
    absolute values of the reference data.

    Parameters
    ----------
    data : numpy array
        Array of data to compare to the reference. Size and shape must match
        reference_data
    reference_data : numpy array
        Array of reference data. Size and shape must match data.

    Returns
    -------
    float
        RMSE of the data relative to the mean absolute value of the reference data.
    """

    errorField = data - reference_data
    rmse_value = rmse(errorField)
    reference_abs_mean = np.mean(abs(reference_data))
    frmse = rmse_value / reference_abs_mean

    return frmse


def parametric_fit(func, x, y, initial_params=None, xmin=None, xmax=None):
    """Fit a function f(x) with some unknown parameters to (x, y) data given a set of
    initial parameters. A simple wrapper of scipy.optimise.curve_fit

    Parameters
    ----------
    func : class 'function'
        A function f(x), with x as the first argument and all successive arguments as
        function parameters to be optimised.
    x : array_like
        Values of the independent variable.
    y : array_like
        Values of the dependent variable corresponding to each point in x.
    initial_params : array_like, optional
        Initial guess for the parameters (length N), by default None.
    xmin : float, optional
        If defined, fit only to values for x>=xmin, by default None.
    xmax : float, optional
        If defined, fit only to values for x<=xmax, by default None.

    Returns
    -------
    popt : array
        Optimal values for the parameters.
    fit_uncertainty : array
        One standard deviation error on each of the parameters.
    """

    if xmin:
        xmask_lower = xmin <= x
        x = x[xmask_lower]
        y = y[xmask_lower]
    if xmax:
        xmask_upper = x <= xmax
        x = x[xmask_upper]
        y = y[xmask_upper]

    popt, pcov = curve_fit(func, x, y, p0=initial_params)
    fit_uncertainty = np.sqrt(np.diag(pcov))
    return popt, fit_uncertainty


def linear_function(x, m, c):
    return m * x + c
