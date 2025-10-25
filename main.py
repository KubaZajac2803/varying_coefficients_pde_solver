from geometry import Square2D
from renderer import MonteCarloPDE2D
from display import plot
import numpy as np
import sympy as sym

np.random.seed(42)
bdr_max = 20
num_walks = 1
epsilon = 10e-3
max_walk_length = 100
methods = {0: "next_flight", 1: "delta_tracking_recursion", 2: "background_values", 3: "screening_coefficient", 4: "diffusion",
           5: "laplacian_diffusion", 6: "norm_gradient_log_diffusion"}
method = methods[0]

def bdr_cond(x):
    if x[0] > bdr_max/2:
        return 1
    else:
        return 0

def source_contribution(x):
    if bdr_max/4 < x[0] < 3*bdr_max/4 and bdr_max / 4 < x[1] < 3 * bdr_max / 4:
        return 0.05
    else:
        return 0

screening_scaling = 1
min_screening = 0.01
sigma_bar = screening_scaling - min_screening
def screening_coefficient(x):
    if bdr_max/4 < x[1] < 3*bdr_max/4:
        return screening_scaling+min_screening
    else:
        return min_screening

stddev = 5
x, y = sym.symbols('x y')
diffusion = 1#2 - sym.exp(-((x-bdr_max/2)**2 + (y-bdr_max/2)**2)/(2*stddev**2))
diffusion_coefficient = sym.lambdify((x, y), diffusion)
laplacian_diffusion = sym.lambdify((x, y), sym.simplify(sym.diff(diffusion, x, x) + sym.diff(diffusion, y, y)))
norm_gradient_log_diffusion = sym.lambdify((x, y), sym.sqrt(sym.simplify(sym.diff(sym.log(diffusion), x)**2 +
                                                                    sym.diff(sym.log(diffusion), y)**2)))

if __name__ == '__main__':
    geometry = Square2D(bdr_max, bdr_cond, source_contribution)
    renderer = MonteCarloPDE2D(geometry, num_walks, epsilon, max_walk_length, method, diffusion_coefficient,
                               laplacian_diffusion, norm_gradient_log_diffusion, screening_coefficient, sigma_bar)
    values = renderer.find_pde()

    plot(values, geometry)