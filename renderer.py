import numpy as np
from scipy import special
from scipy import integrate

class MonteCarloPDE2D:
    def __init__(self, geometry, num_walks, epsilon, max_walk_length, method, diffusion=lambda x: 0, diffusion_laplacian = lambda x:0, screening_coeff=lambda x: 0, max_screening = 0):
        self.geometry = geometry
        self.num_walks = num_walks
        self.epsilon = epsilon
        self.max_walk_length = max_walk_length
        self.points_to_check = geometry.points_to_check()
        self.method = method
        self.diffusion = diffusion
        self.diffusion_laplacian = diffusion_laplacian
        self.screening_coeff = screening_coeff
        self.max_screening = max_screening
        self.hash_length = 1000

    def Greens_2D(self, r, ball_radius, max_screening):
         return (1/(2*np.pi))*(special.k0(r * np.sqrt(max_screening))
                    - (special.i0(r * np.sqrt(max_screening))
                    * special.k0(ball_radius * np.sqrt(max_screening))
                    / special.i0(ball_radius * np.sqrt(max_screening))))

    def Greens_2D_integral(self, ball_radius):
        return (1/self.max_screening)*(1 - 1/(special.i0(ball_radius * np.sqrt(self.max_screening))))

    def local_screening(self, source_point):
        return (self.screening_coeff(source_point) / self.diffusion(source_point) +
                (self.diffusion_laplacian(source_point)/self.diffusion(source_point) - (abs(np.log(self.diffusion(source_point)))**2)/2) / 2)

    def CDF(self, Greens_2D, hash_length, max_screening):
        x = np.linspace(1/hash_length, 1, hash_length)
        y = np.array([Greens_2D(i, 1, max_screening) for i in x])
        cdf_hash = np.zeros(hash_length)
        cdf_hash[0] = 0
        for i in range(1, len(x)-1):
            cdf_hash[i] = cdf_hash[i-1] + (y[i] + y[i-1])/(2*hash_length) #trapezoid integration method + memoization
        cdf_hash = np.array(cdf_hash)
        return cdf_hash/np.linalg.norm(cdf_hash)


    #recursion_level = 0
    def delta_tracking(self, point_to_check, epsilon, max_walk_length, diffusion, screening_coeff, max_screening, Greens_2D, Greens_2D_integral, CDF_values):
        point_to_check = np.array(point_to_check)
        closest_boundary_point = self.geometry.closest_boundary_point(point_to_check)
        ball_radius = np.linalg.norm(point_to_check - closest_boundary_point)
        if ball_radius <= epsilon or max_walk_length <= 0:
            return self.geometry.value_at_boundary(closest_boundary_point)
        else:
            mu = np.random.random()
            rand_radius = ball_radius * CDF_values[np.random.randint(0, 999)] #the cdf is the problem
            rand_angle_2 = 2 * np.pi * np.random.random()
            source_point = point_to_check + rand_radius * np.array([np.cos(rand_angle_2), np.sin(rand_angle_2)])
            source_term = Greens_2D_integral(rand_radius)/(np.sqrt(diffusion(point_to_check) * diffusion(source_point)))*self.geometry.value_at_background(source_point)
            if mu <= max_screening*Greens_2D_integral(ball_radius):
                sigma_prime = self.local_screening(source_point)
                return (source_term + np.sqrt(diffusion(source_point)/diffusion(point_to_check))*(1-sigma_prime/max_screening)*
                        self.delta_tracking(source_point, epsilon, max_walk_length-1, diffusion, screening_coeff, max_screening, Greens_2D, Greens_2D_integral, CDF_values))
            else:
                rand_angle_1 = 2 * np.pi * np.random.random()
                new_point = point_to_check + ball_radius * np.array([np.cos(rand_angle_1), np.sin(rand_angle_1)])
                return (source_term + np.sqrt(diffusion(new_point)/diffusion(point_to_check))*
                        self.delta_tracking(new_point, epsilon, max_walk_length-1, diffusion, screening_coeff, max_screening, Greens_2D, Greens_2D_integral, CDF_values))


    def find_pde(self):
        result = np.zeros(len(self.points_to_check))
        #derivatives = []
        CDF_hash = self.CDF(self.Greens_2D, self.hash_length, self.max_screening)
        match self.method:
            case "delta_tracking":
                for i, point_to_check in enumerate(self.points_to_check):
                    for walk_num in range(self.num_walks):
                        result[i] += self.delta_tracking(point_to_check, self.epsilon, self.max_walk_length, self.diffusion, self.screening_coeff, self.max_screening, self.Greens_2D, self.Greens_2D_integral, CDF_hash)/self.num_walks
            case _:
                pass