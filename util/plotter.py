import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_phase(ax):
    x = [-4, 4]; y1 = [-4, 4]; y2 = [4, -4]
    ax.plot(x, y1, color = "grey", linewidth = 1, alpha =  0.5); ax.plot(x, y2, color = "grey", linewidth = 1, alpha = 0.5)
    ax.axhline(0, color = "grey", linewidth = 1, alpha = 0.5); ax.axvline(0, color = "grey", linewidth = 1, alpha = 0.5)
    circle = patches.Circle((0, 0), 1, edgecolor = "grey", linewidth = 1, facecolor = "none", alpha = 0.5)
    ax.add_patch(circle)