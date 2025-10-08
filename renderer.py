import numpy as np
from scipy import special
from scipy import stats

class MonteCarloPDE2D:
    def __init__(self, geometry, num_walks, epsilon, max_walk_length, method, diffusion=lambda x: 0, screening_coeff=lambda x: 0, max_screening = 0):
        self.geometry = geometry
        self.num_walks = num_walks
        self.epsilon = epsilon
        self.max_walk_length = max_walk_length
        self.points_to_check = geometry.points_to_check()
        self.method = method
        self.diffusion = diffusion
        self.screening_coeff = screening_coeff
        self.max_screening = max_screening

    def Greens_2D(self, r, ball_radius, max_screening):
         return (1/(2*np.pi))*(special.k0(r * np.sqrt(max_screening))
                    - (special.i0(r * np.sqrt(max_screening))
                    * special.k0(ball_radius * np.sqrt(max_screening))
                    / special.i0(ball_radius * np.sqrt(max_screening))))

    def Greens_2D_integral(self, ball_radius, max_screening):
        return (1/max_screening)*(1 - 1/(special.i0(ball_radius * np.sqrt(max_screening))))

    """def CDF(self, r):
        return self.Greens_2D(r, 1, self.max_screening)/self.Greens_2D_integral(1, self.max_screening)
    CDF_hash = np.array([CDF(x / 1000) for x in range(1, 1001)])""" #this is wrong as hell

    def delta_tracking(self, point_to_check, epsilon, max_walk_length, diffusion, screening_coeff, max_screening, Greens_2D, Greens_2D_integral, CDF_values):
        point_to_check = np.array(point_to_check)
        closest_boundary_point = self.geometry.closest_boundary_point(point_to_check)
        ball_radius = np.linalg.norm(point_to_check - closest_boundary_point)
        if ball_radius <= epsilon:
            return self.geometry.value_at_boundary(closest_boundary_point)
        else:
            mu = np.random.random()
            rand_angle_1 = 2*np.pi*np.random.random()
            new_point = point_to_check + ball_radius*np.array([np.cos(rand_angle_1), np.sin(rand_angle_1)])
            if mu <= max_screening*Greens_2D_integral(ball_radius, max_screening):
                rand_radius = ball_radius * CDF_values[np.random.randint(0, 999)]
                rand_angle_2 = 2*np.pi*np.random.random()
                source_point = point_to_check + rand_radius*np.array([np.cos(rand_angle_1), np.sin(rand_angle_1)])
                pass
            else:
                return np.sqrt(diffusion(new_point)/diffusion(point_to_check))*self.delta_tracking(new_point, epsilon, max_walk_length-1, diffusion, screening_coeff, max_screening, Greens_2D, Greens_2D_integral)


    def find_pde(self):
        result = []
        derivatives = []
        match self.method:
            case "delta_tracking":
                pass
            case _:
                pass