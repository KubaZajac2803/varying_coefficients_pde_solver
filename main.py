from geometry import Square2D
from renderer import MonteCarloPDE2D
from display import Display
import matplotlib.pyplot as plt
import numpy as np
import datetime


bdr_max = 20
num_walks = 20
epsilon = 10e-6
max_walk_length = 100
method = "delta_tracking"
btm_left = [0,0]

def bdr_cond(x):
    if x[0]>bdr_max/2:
        return 1
    else:
        return 0

def source_contribution(x):
    if x[0] > bdr_max/4 and x[0] < 3*bdr_max/4 and x[1] < bdr_max/2 :
        return 1
    else:
        return 0

screening_scaling = 0.5
min_screening = 0.01
sigma_bar = screening_scaling - min_screening
def screening_coefficient(x):
    if x[0] > bdr_max/4 and x[0] < 3*bdr_max/4 and x[1] < bdr_max/2 :
        return screening_scaling+min_screening
    else:
        return min_screening

stddev = 5
def diffusion_coefficient(x):
    return 1/(2*np.pi*stddev**2)*np.exp(-((x[0]-bdr_max/2)**2 + (x[1]-bdr_max/2)**2)/(2*stddev**2))+0.01

def diffusion_laplacian(x):
    c = bdr_max / 2
    A = 1 / (2 * np.pi * stddev ** 2)
    r2 = (x[0] - c) ** 2 + (x[1] - c) ** 2
    exp_term = np.exp(-r2 / (2 * stddev ** 2))
    return A * exp_term * ((r2 - 2 * stddev ** 2) / stddev ** 4)

if __name__ == '__main__':
    square_geometry = Square2D(btm_left[0],btm_left[1], bdr_max, bdr_cond, source_contribution)
    renderer = MonteCarloPDE2D(square_geometry, num_walks, epsilon, max_walk_length, method, diffusion_coefficient,
                               diffusion_laplacian, screening_coefficient, sigma_bar)
    values = renderer.find_pde()
    plotter = Display(values)
    plotter.plot()
