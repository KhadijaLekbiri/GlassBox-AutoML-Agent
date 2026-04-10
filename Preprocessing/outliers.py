import numpy as np

def detect_outliers(data):
    """
    Detect outliers in the dataset using the IQR method.

    Parameters:
    - data: Input array of data.

    Returns:
    - outliers: Array of outlier data points.
    - outlier_indices: Indices of outlier data points.
    """
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))
    outliers = data[outlier_indices]

    return outliers, outlier_indices