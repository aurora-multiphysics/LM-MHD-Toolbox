import numpy as np


def file_read_cartesian_2D(
    file,
    shape,
    xCol,
    yCol,
    dataCols,
    coord_tol=1e-5,
    delimiter=",",
    skiprows=0,
    shape_order="xy",
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
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
    shape_order : str, optional
        If "xy" (default) shape is (x length, y length), if "yx" shape is (y length, x length)
    xmin : float, optional
        If defined, discard all points where x < xmin
    xmax : float, optional
        If defined, discard all points where x > xmax
    ymin : float, optional
        If defined, discard all points where y < ymin
    ymax : float, optional
        If defined, discard all points where y > ymax


    Returns
    -------
    x : list of float
        1D list of x coordinates
    y : list of float
        1D list of y coordinates
    data : array of float, or list of array of float
        2D data arrays
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

    # optionally mask out parts of the domain
    row_ids = range(np.shape(full_data)[0])
    delete_list = []
    for row_id in row_ids:
        x = full_data[row_id, xCol]
        y = full_data[row_id, yCol]
        if (
            (xmin and x < xmin)
            or (xmax and x > xmax)
            or (ymin and y < ymin)
            or (ymax and y > ymax)
        ):
            delete_list.append(row_id)
    full_data = np.delete(full_data, delete_list, axis=0)

    # sort by x, then sort by y maintaining previous sorting
    full_data = full_data[full_data[:, xCol].argsort()]
    full_data = full_data[full_data[:, yCol].argsort(kind="mergesort")]

    # extract 1D arrays of all points from specified columns
    x_data = full_data[:, xCol]
    y_data = full_data[:, yCol]

    # reshape into 2D grids (rows with varying x, columns with varying y)
    if shape_order == "xy":
        yx_shape = shape[::-1] # reverse the shape
    elif shape_order == "yx":
        yx_shape = shape
    else:
        raise Exception('shape order must be "xy" or "yx"')
    x_data = np.reshape(x_data, yx_shape)
    y_data = np.reshape(y_data, yx_shape)

    # grid is regular cartesian so can work with just 1D arrays
    x = x_data[0]
    y = y_data[:, 0]

    # get the data
    if dataNumCols == 1:
        data = full_data[:, dataCols]
        data = np.reshape(data, yx_shape)
    else:
        data = []
        for i in range(dataNumCols):
            data_column = full_data[:, dataCols[i]]
            data_column = np.reshape(data_column, yx_shape)
            data.append([])
            data[i] = data_column

    return x, y, data
