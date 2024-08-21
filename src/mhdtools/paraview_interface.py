import numpy as np


def file_read_cartesian_2D(
    file, shape, xCol, yCol, dataCols, coord_tol=1e-5, delimiter=",", skiprows=0
):
    """Read a 2D slice with a structured cartesian grid from file e.g. .csv,
    and reshape it into a 2D grid in terms, returning x, y and data lists, where
    data is either a list of scalars or a list of lists of scalars.

    Parameters
    ----------
    file : str
        Name of file to read
    shape : tuple (int, int)
        _description_
    xCol : int
        Column number of x coordinates
    yCol : int
        Column number of y coordinates
    dataCols : int or list of int
        Column number for data fields to extract, or list of column numbers for multiple fields.
    coord_tol : float, optional
        Upper limit below which values near zero to set to exactly zero, by default 1e-5
    delimiter : str, optional
        File column delimiter, by default ","
    skiprows : int, optional
        Number of rows to skip when reading file, by default 0

    Returns
    -------
    x : list of float
        1D list of x coordinates
    y : list of float
        1D list of y coordinates
    data : list of float, or list of list of float
        Non-coordinate data lists
    """

    if type(dataCols) is not list:
        dataNumCols = 1
    else:
        dataNumCols = len(dataCols)

    full_data = np.loadtxt(file, skiprows=1, delimiter=",")

    # some coordinates are near zero but not quite; messes with ordering
    # this step sets these to zero so that ordering is correct
    full_data[:, xCol][abs(full_data[:, xCol]) < coord_tol] = 0
    full_data[:, yCol][abs(full_data[:, yCol]) < coord_tol] = 0

    # sort by x, then sort by y maintaining previous sorting
    full_data = full_data[full_data[:, xCol].argsort()]
    full_data = full_data[full_data[:, yCol].argsort(kind="mergesort")]

    # extract 1D arrays from specified columns
    x_data = full_data[:, xCol]
    y_data = full_data[:, yCol]

    # reshape into 2D grids
    x_data = np.reshape(x_data, shape)
    y_data = np.reshape(y_data, shape)

    # grid is regular so can work with just 1D arrays
    y = y_data[:, 0]
    x = x_data[0]

    # get the data
    if dataNumCols == 1:
        data = full_data[:, dataCols]
        data = np.reshape(data, shape)
        data = np.transpose(data)
    else:
        data = []
        for i in range(dataNumCols):
            data_column = full_data[:, dataCols[i]]
            data_column = np.reshape(data_column, shape)
            data.append([])
            data[i] = np.transpose(data_column)

    return x, y, data
