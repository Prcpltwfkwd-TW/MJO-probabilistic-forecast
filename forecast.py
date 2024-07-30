# %% Packages
import json
import numpy as np
from scipy.linalg import expm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# %%
from util.forecast_tool import generate_init_condition, find_coverage_position, smooth_process, noise_at_each_timestep
from util.plotter import plot_phase

# %% Main

# Read data
M     = np.load("M.npy") # MJO data
dM_dt = np.roll(M, -1, axis = 1) - M
G     = expm(np.dot(np.dot(dM_dt, M.T), np.linalg.inv(np.dot(M, M.T))))

cov = np.load("noise.npy") # state-dependent covariance matrix
det = np.load("det.npy") # state-dependent determinant

# Model configuration
x_min = -4; x_max = 4; y_min = -4; y_max = 4 # boundary of the phase space
gridsnumber = cov.shape[0]; det_gridsnumber = det.shape[0]

x_det  = y_det  = np.linspace(x_min, x_max, det_gridsnumber)
X_det,  Y_det   = np.meshgrid(x_det, y_det)
x_grid = y_grid = np.linspace(x_min, x_max, gridsnumber)
X_grid, Y_grid  = np.meshgrid(x_grid, y_grid)

n_ens      = 200 # number of particles to resolve the PDF
t_forecast = 250 # forecast length
particles_forecast = np.zeros((n_ens, 2, t_forecast + 1))
rho_forecast       = np.zeros((n_ens, t_forecast + 1))

# Initial condition
x0, y0 = 1, 0 # center of the initial condition
cov0   = [[0.05, 0], [0, 0.05]] # covariance matrix of the initial condition
rho_init  = generate_init_condition([x0, y0], cov0, M)[:-t_forecast]
rho_init  = smooth_process(rho_init, M[0, :-t_forecast], M[1, :-t_forecast], X_grid, Y_grid)
rho_init /= np.nansum(rho_init)
dict_95_init = find_coverage_position(rho_init, 0.95)
particles_init = np.random.choice(dict_95_init["positions"], n_ens, replace = False) # particles used to resolve the PDF

particles_forecast[:, :, 0] = M[:, particles_init].T
rho_forecast[:, 0]          = rho_init[particles_init]
rho_forecast[:, 0]         /= np.nansum(rho_forecast[:, 0])

# Forecast
for t in range(1, t_forecast):
    
    # predict the particles
    for ens in range(n_ens):
        xposition = np.argmin(np.abs(x_det - particles_forecast[ens, 0, t-1])); yposition = np.argmin(np.abs(y_det - particles_forecast[ens, 1, t-1]))
        epsilon   = np.random.multivariate_normal([0, 0], cov[yposition, xposition], 1)
        particles_forecast[ens, :, t] = np.dot(G, particles_forecast[ens, :, t-1]) + epsilon
    
    # predict the PDF
    det_model = griddata((X_det.flatten(), Y_det.flatten()), det.flatten(), (particles_forecast[:, 0, t], particles_forecast[:, 1, t]), method = "nearest")
    rho_forecast[:, t]  = rho_forecast[:, t-1] / det_model
    rho_forecast[:, t]  = smooth_process(rho_forecast[:, t], particles_forecast[:, 0, t], particles_forecast[:, 1, t], X_grid, Y_grid)
    rho_forecast[:, t] /= np.nansum(rho_forecast[:, t])

# Output the data
np.save("particles_forecast.npy", particles_forecast)
np.save("rho_forecast.npy", rho_forecast)

# %% Plot
fig = plt.figure(figsize = (12, 16), dpi = 300)
gs  = GridSpec(4, 4, figure = fig)

count = 0
for i in range(4):
    for j in range(4):
        t = count * 5
        ax = fig.add_subplot(gs[i, j])
        ax.set_aspect("equal", adjustable = "box")
        rho_plot_gridded = griddata((particles_forecast[:, 0, t], particles_forecast[:, 1, t]), rho_forecast[:, t], (X_grid, Y_grid), method = "linear")
        dict_95          = find_coverage_position(rho_forecast[:, t], 0.95)
        
        axrho = ax.contourf(X_grid, Y_grid, rho_plot_gridded, cmap = "Greys", extend = "max", levels = np.arange(0, 0.014+1e-8, 0.001))
        ax.contour(X_grid, Y_grid, rho_plot_gridded, levels = [dict_95["value"]], colors = "red")

        ax.scatter(particles_forecast[:, 0, t], particles_forecast[:, 1, t], color = "gold", edgecolor = "k", s = 10)
        center = np.nanargmax(rho_plot_gridded)
        ax.scatter(X_grid.flatten()[center], Y_grid.flatten()[center], color = "blue", edgecolor = "k", s = 15)
        plot_phase(ax)
        
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_title("t = %d" % t, fontsize = 14)
        count += 1

cbox = fig.add_axes([gs[3, 3].get_position(fig).x1 + 0.02, gs[3, 3].get_position(fig).y0, 0.03, gs[0, 3].get_position(fig).y1 - gs[3, 3].get_position(fig).y0])
cbar = plt.colorbar(axrho, cax = cbox, orientation = "vertical")

plt.suptitle("LIM-based Liouville Equation Model", fontsize = 20, y = 0.9)
plt.savefig("images/forecast.png", dpi = 300)