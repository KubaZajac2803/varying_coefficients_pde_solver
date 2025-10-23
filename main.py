from geometry import Square2D
from renderer import MonteCarloPDE2D
from display import plot
import numpy as np
import sympy as sym


bdr_max = 20
num_walks = 5
epsilon = 10e-6
max_walk_length = 100
methods = ["delta_tracking_recursion", "diffusion", "screening_coefficient", ]
method = methods[0]
btm_left = [0, 0]

def bdr_cond(x):
    if x[0] > bdr_max/2:
        return 1
    else:
        return 0.1

def source_contribution(x):
    if x[0] > bdr_max/4 and x[0] < bdr_max/2:
        return 0.2
    else:
        return 0

screening_scaling = 1
min_screening = 0.01
sigma_bar = screening_scaling - min_screening
def screening_coefficient(x):
    if x[1] >= (x[0]-bdr_max/2)**2 and x[1] <= (x[0]-bdr_max/2)**2 + 4: #visible parabola where screening is much higher
        return screening_scaling+min_screening
    else:
        return min_screening

stddev = 5
x, y = sym.symbols('x y')
diffusion = 1  #160*(1/(2*sym.pi*stddev**2)*sym.exp(-((x-bdr_max/2)**2 + (y-bdr_max/2)**2)/(2*stddev**2)))
diffusion_coefficient = sym.lambdify((x, y), diffusion)
laplacian_diffusion = sym.lambdify((x, y), sym.simplify(sym.diff(diffusion, x, x) + sym.diff(diffusion, y, y)))
laplacian_log_diffusion = sym.lambdify((x, y), sym.Abs(sym.simplify(sym.diff(sym.log(diffusion), x, x) +
                                                                    sym.diff(sym.log(diffusion), y, y))))

if __name__ == '__main__':
    geometry = Square2D(btm_left[0], btm_left[1], bdr_max, bdr_cond, source_contribution)
    renderer = MonteCarloPDE2D(geometry, num_walks, epsilon, max_walk_length, method, diffusion_coefficient,
                               laplacian_diffusion, laplacian_log_diffusion, screening_coefficient, sigma_bar)
    values = renderer.find_pde()
    plot(values, geometry)