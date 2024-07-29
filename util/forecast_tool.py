# %% Packages
import numpy as np
import scipy.stats as stats
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# %% Functions
def convert_data_to_2d_pdf(x, y, ngrid):
    """
    Convert the given 2D data to 2D PDF.
    
    Parameters:
    -----------
        x: np.ndarray
            the first dimension of the data
        
        y: np.ndarray
            the second dimension of the data
        
        ngrid: int
            the number of grid points that the output PDF will have in one dimension
        
        bound_factor (int, optional): _description_. Defaults to 10.

    Returns:
    --------
        X: np.ndarray
            the position of the grid points in the first dimension
            
        Y: np.ndarray
            the position of the grid points in the second dimension
        
        rho: np.ndarray
            the 2D PDF of the data
    """
    xmin = -4; xmax = 4; ymin = -4; ymax = 4
    X, Y      = np.mgrid[xmin:xmax:ngrid*1j, ymin:ymax:ngrid*1j]
    positions = np.vstack([X.flatten(), Y.flatten()])
    values    = np.vstack([x, y])
    kernel    = stats.gaussian_kde(values)
    rho       = np.reshape(kernel(positions).T, X.shape)
    rho      /= np.trapz(np.trapz(rho, X[:, 0], axis = 0), Y[0, :], axis = 0)
    return X, Y, rho

def calc_KL_div(p, q, gridsize = 0.02):
    """
    Calculate the Kullback-Leibler divergence between two PDFs.
    
    Parameters:
    -----------
        p: np.ndarray
            the target PDF
        
        q: np.ndarray
            the reference PDF
        
        gridsize: float, optional
            the grid size of the PDFs. Defaults to 0.02.
            
    Returns:
    --------
        KL_div: float
            the Kullback-Leibler divergence between the two PDFs
    """
    return (p * np.log(p / q) * gridsize**2).sum()

def generate_init_condition(center: np.ndarray, cov: np.ndarray, points: np.ndarray):
    """
    Generate the initial condition based on the given center and covariance matrix.

    Parameters:
    -----------
        center: np.ndarray
            the center of the multivariate normal distribution
        cov: np.ndarray
            the covariance matrix of the multivariate normal distribution
        points: np.ndarray
            the grid points of the PDF

    Returns:
    --------
        rho: np.ndarray
            the initial condition based on the given center and covariance matrix
    """
    rv  = stats.multivariate_normal(center, cov)
    rho = np.zeros(points.shape[1])
    for i in range(points.shape[1]):
        rho[i] = rv.pdf([points[0, i], points[1, i]])
    return rho / np.nansum(rho)

def _calc_sum_of_pdf(rho, percentage):
    """
    Calculate the sum of the PDF based on the given percentage.
    
    Parameters:
    -----------
        rho: np.ndarray
            the input PDF
        
        percentage: float
            the percentage of the PDF that we want to calculate the sum of

    Raises:
    -------
        ValueError: If the percentage is not between 0 and 100

    Returns:
    --------
        value_less: float
            the value of the PDF that is less than the given percentile
        
        value_more: float
            the value of the PDF that is more than the given percentile
        
        posi_less: np.ndarray
            the positions of the PDF that are less than the given percentile
            
        posi_more: np.ndarray
            the positions of the PDF that are more than the given percentile
            
        sum_less: float
            the sum of the PDF that are less than the given percentile
            
        sum_more: float
            the sum of the PDF that are more than the given percentile
    """
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage should be between 0 and 100")
    
    rho_normalized = rho / np.nansum(rho) # normalize the PDF
    value_less = np.nanpercentile(rho_normalized, 100 - percentage - 1); value_more = np.nanpercentile(rho_normalized, 100 - percentage)
    posi_less  = np.where(rho_normalized >= value_less)[0];              posi_more  = np.where(rho_normalized >= value_more)[0]
    sum_less   = np.nansum(rho[posi_less]);                              sum_more   = np.nansum(rho[posi_more])
    
    return value_less, value_more, posi_less, posi_more, sum_less, sum_more

def find_coverage_position(rho, coverage = 0.95):
    """
    Find the position of the PDF that covers the given percentage.
    
    Parameters:
    -----------
        rho: np.ndarray
            the input PDF
        coverage: float, optional
            the percentage of the PDF that we want to cover. Defaults to 0.95.

    Returns:
        dict_coverage: dict
            a dictionary that contains the coverage percentage, the value of the PDF, the positions of the PDF, and the sum of the PDF
    """
    for i in range(100):
        value_less, value_more, posi_less, posi_more, sum_less, sum_more = _calc_sum_of_pdf(rho, i)
        if (sum_less - coverage) * (sum_more - coverage) < 0:
            dict_coverage = {"coverage": coverage, "value": (value_more + value_less) / 2, "positions": posi_more, "sum": sum_more}
            return dict_coverage

def smooth_process(rho: np.ndarray, m1: np.ndarray, m2: np.ndarray, X_grid: np.ndarray, Y_grid: np.ndarray, gaussian_sigma: float = 20):
    """
    Smooth the given PDF. The dimension of input and output PDFs are the same.
    
    Parameters:
    -----------
        rho: np.ndarray
            the input PDF
        
        m1: np.ndarray
            the first dimension of the input PDF
            
        m2: np.ndarray
            the second dimension of the input PDF
        
        X_grid: np.ndarray
            the grid points of the first dimension
        
        Y_grid: np.ndarray
            the grid points of the second dimension
        
        gaussian_sigma: float, optional
            the sigma of the Gaussian filter. Defaults to 20.

    Returns:
    --------
        rho_new: np.ndarray
            the smoothed PDF
    """
    rho_gridded = griddata((m1, m2), rho, (X_grid, Y_grid), method = "linear", fill_value = 1e-36)
    rho_gridded = gaussian_filter(rho_gridded, sigma = gaussian_sigma)
    rho_new  = griddata((X_grid.flatten(), Y_grid.flatten()), rho_gridded.flatten(), (m1, m2), method = "nearest")
    return rho_new


def noise_at_each_timestep(cov, size = 1):
    """
    Generate the noise based on the given covariance matrix.
    
    Parameters:
    -----------
        cov: np.ndarray
            the covariance matrix of the noise
        size: int, optional
            the number of noise to generate. Defaults to 1.

    Returns:
    --------
        noise: np.ndarray
            the generated noise
    """
    noise = np.random.multivariate_normal([0, 0], cov, size = size)
    return noise.T