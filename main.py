from geometry import Square2D, Rectangle2D, ParametricHalfSphere
from renderer import MonteCarloPDE2D, EuclideanBrownianMotion, RiemannianBrownianMotion, MonteCarloRiemannianPDE2D
from display import heatmap, graph_flat_walk, graph_walk_on_surface, heatmap_riemannian
import numpy as np
import sympy as sym

np.random.seed(42)
bdr_max_x = 2*np.pi
bdr_max_y = np.pi/2
num_walks = 300
epsilon = 10e-3
max_walk_length = 5000
methods = {0: "next_flight", 1: "delta_tracking_recursion", 2: "background_values", 3: "screening_coefficient",
           4: "diffusion", 5: "laplacian_diffusion", 6: "norm_gradient_log_diffusion"}
method = methods[4]


def bdr_cond(x):
    if x[0] > bdr_max_x/2:
        return 1
    else:
        return 0


def source_contribution(x):
    if bdr_max_x/4 < x[0] < 3*bdr_max_x/4 and bdr_max_y / 4 < x[1] < 3 * bdr_max_y / 4:
        return 0.05
    else:
        return 0


screening_scaling = 1
min_screening = 0.01
sigma_bar = screening_scaling - min_screening
def screening_coefficient(x):
    if bdr_max_y/4 < x[1] < 3*bdr_max_y/4:
        return screening_scaling + min_screening
    else:
        return min_screening


stddev = 2
x, y = sym.symbols('x y')
diffusion = 1# + sym.exp(-((x-bdr_max_x/2)**2 + (y-bdr_max_y/2)**2)/(2*stddev**2))
diffusion_coefficient = sym.lambdify((x, y), diffusion)
laplacian_diffusion = sym.lambdify((x, y), sym.simplify(sym.diff(diffusion, x, x) + sym.diff(diffusion, y, y)))
norm_gradient_log_diffusion = sym.lambdify((x, y), sym.sqrt(sym.simplify(sym.diff(sym.log(diffusion), x)**2 +
                                                                    sym.diff(sym.log(diffusion), y)**2)))

u, v = sym.var('u v')
surface_parameterization = sym.Matrix([sym.cos(u)*sym.cos(v), sym.sin(u)*sym.cos(v), sym.sin(v)])

time_step = 10e-4

number_of_samples = 50

if __name__ == '__main__':

    half_sphere = ParametricHalfSphere(number_of_samples)
    """
    geometry = Rectangle2D(bdr_max_x, bdr_max_y, bdr_cond, source_contribution,
                           np.array([[[0, 0], [bdr_max_x, 0]]]))
    
    renderer = MonteCarloPDE2D(geometry, num_walks, epsilon, max_walk_length, method, diffusion_coefficient,
                               laplacian_diffusion, norm_gradient_log_diffusion, screening_coefficient, sigma_bar)
    values = renderer.find_pde()
    heatmap(values, geometry)
    
    for n in range(2):
        starting_point = [0, np.pi/2]
        renderer = RiemannianBrownianMotion(geometry, 2e-10, starting_point, max_walk_length, diffusion_coefficient, time_step,
                                            surface_parameterization)

        positions = renderer.perform_walk(split=True)
        graph_flat_walk(positions, geometry)
        graph_walk_on_surface(positions, geometry, surface_parameterization)
        print(n)
    """
    renderer = MonteCarloRiemannianPDE2D(half_sphere, num_walks, epsilon, max_walk_length, diffusion_coefficient,
                                         time_step, surface_parameterization)
    values = renderer.find_pde()
    heatmap_riemannian(values, half_sphere, surface_parameterization)