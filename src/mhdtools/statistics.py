import numpy as np

# from scipy.optimize import curve_fit
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
