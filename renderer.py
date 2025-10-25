import numpy as np
from scipy import special
import copy
import time

class MonteCarloPDE2D:
    def __init__(self, geometry, num_walks, epsilon, max_walk_length, method, diffusion=lambda x: 0, laplacian_diffusion = lambda x:0, norm_gradient_log_diffusion = lambda x:0, screening_coeff=lambda x: 0, max_screening = 0):
        self.geometry = geometry
        self.num_walks = num_walks
        self.epsilon = epsilon
        self.max_walk_length = max_walk_length
        self.points_to_check = geometry.points_to_check()
        self.method = method
        self.diffusion = diffusion
        self.laplacian_diffusion = laplacian_diffusion
        self.norm_gradient_log_diffusion = norm_gradient_log_diffusion
        self.screening_coeff = screening_coeff
        self.max_screening = max_screening
        self.hash_length = 1000

    def Greens_2D(self, ball_radius, r):
         return (1/(2*np.pi))*(special.k0(r * np.sqrt(self.max_screening))
                    - (special.i0(r * np.sqrt(self.max_screening))
                    * special.k0(ball_radius * np.sqrt(self.max_screening))
                    / special.i0(ball_radius * np.sqrt(self.max_screening))))

    def Greens_2D_integral(self, ball_radius):
        return (1/self.max_screening)*(1 - 1/(special.i0(ball_radius * np.sqrt(self.max_screening))))

    def sigma_prime(self, inside_point):
        return (self.screening_coeff(inside_point) / self.diffusion(*inside_point) +
                (self.laplacian_diffusion(*inside_point)/self.diffusion(*inside_point) -
                 (self.norm_gradient_log_diffusion(*inside_point)**2)/2) / 2)

    def CDF(self):
        x = np.linspace(1/self.hash_length, 1, self.hash_length)
        y = np.array([self.Greens_2D(1, i) for i in x])
        cdf_hash = np.zeros(self.hash_length)
        cdf_hash[0] = 0
        for i in range(1, len(x)-1):
            cdf_hash[i] = cdf_hash[i-1] + (y[i] + y[i-1])/(2*self.hash_length) #trapezoid integration method
        cdf_hash = np.array(cdf_hash)
        return cdf_hash/np.sum(cdf_hash)

    def delta_tracking_recursion(self, point_to_check, max_walk_length, CDF_values):
        point_to_check = np.array(point_to_check)
        closest_boundary_point = self.geometry.closest_boundary_point(point_to_check)
        ball_radius = np.linalg.norm(point_to_check - closest_boundary_point)

        if ball_radius <= self.epsilon or max_walk_length <= 0:
            return self.geometry.value_at_boundary(closest_boundary_point)
        else:
            mu = np.random.random()
            rand_radius = ball_radius * CDF_values[np.random.randint(1, self.hash_length - 1)]
            rand_angle_2 = 2 * np.pi * np.random.random()
            inside_point = point_to_check + rand_radius * np.array([np.cos(rand_angle_2), np.sin(rand_angle_2)])
            source_term = self.Greens_2D_integral(rand_radius)/(np.sqrt(self.diffusion(*point_to_check) * self.diffusion(*inside_point)))*self.geometry.value_at_background(inside_point)
            if mu <= self.max_screening*self.Greens_2D_integral(ball_radius):
                sigma_prime = self.sigma_prime(inside_point)
                return source_term + (np.sqrt(self.diffusion(*inside_point)/self.diffusion(*point_to_check))
                        *(1-sigma_prime/self.max_screening)*self.delta_tracking_recursion(inside_point, max_walk_length-1, CDF_values))
            else:
                rand_angle_1 = 2 * np.pi * np.random.random()
                bdr_point = point_to_check + ball_radius * np.array([np.cos(rand_angle_1), np.sin(rand_angle_1)])
                return source_term + (np.sqrt(self.diffusion(*bdr_point)/self.diffusion(*point_to_check)) *
                        self.delta_tracking_recursion(bdr_point, max_walk_length-1, CDF_values))

    def Off_centered_greens_2D(self, ball_radius, center, curr_src_point, new_src_point):
        c = center
        x = curr_src_point
        y = new_src_point
        u = x - c
        v = y - c
        w = y - x
        return (self.Greens_2D(ball_radius, np.linalg.norm(w))
                - self.Greens_2D(ball_radius, (ball_radius**2 - (u@v))/ball_radius))

    def V_2D(self, ball_radius, r):
        return np.sqrt(self.max_screening)*(special.k1(r * np.sqrt(self.max_screening))
                    + (special.i1(r * np.sqrt(self.max_screening))
                    * special.k0(ball_radius * np.sqrt(self.max_screening))
                    / special.i0(ball_radius * np.sqrt(self.max_screening))))

    def Off_centered_Poisson_2D(self, ball_radius, center, off_center_point, bdr_point):
        c = center
        x = off_center_point
        y = bdr_point
        u = x - c
        v = y - c
        w = y - x
        return (1/(2*np.pi))*(self.V_2D(ball_radius, np.linalg.norm(w)) * (v@v - u@v)/(np.linalg.norm(w)*np.linalg.norm(v))
                + self.V_2D(ball_radius, (ball_radius**2 - (u@v))/ball_radius) * (u@v)/(ball_radius*np.linalg.norm(v)))

    def next_flight(self, point_to_check, max_walk_length):
        point_to_check = np.array(point_to_check)
        closest_boundary_point = self.geometry.closest_boundary_point(point_to_check)
        ball_radius = np.linalg.norm(point_to_check - closest_boundary_point)

        if ball_radius <= self.epsilon or max_walk_length <= 0:
            return self.geometry.value_at_boundary(closest_boundary_point)
        else:
            T = 0
            S = 0
            W = 1
            rand_angle_1 = 2 * np.pi * np.random.random()
            bdr_point = point_to_check + ball_radius * np.array([np.cos(rand_angle_1), np.sin(rand_angle_1)])
            curr_sample_point = np.array(copy.deepcopy(point_to_check))
            while True:
                T += (W * self.Off_centered_Poisson_2D(ball_radius, point_to_check, curr_sample_point, bdr_point)
                      / (2*np.pi*ball_radius))
                russian_roulette_probability = min(1, W)
                if russian_roulette_probability < np.random.random():
                    break
                W /= russian_roulette_probability
                rand_radius = ball_radius * np.random.random()
                rand_angle_2 = 2 * np.pi * np.random.random()
                new_sample_point = point_to_check + rand_radius * np.array([np.cos(rand_angle_2), np.sin(rand_angle_2)])
                W *= (self.Off_centered_greens_2D(ball_radius, point_to_check, curr_sample_point, new_sample_point)
                      * (self.max_screening - self.sigma_prime(new_sample_point))*self.Greens_2D_integral(ball_radius)
                      / self.Greens_2D(ball_radius, np.linalg.norm(point_to_check - new_sample_point)))
                S += (self.geometry.value_at_boundary(new_sample_point)*W
                      / (np.sqrt(self.diffusion(*new_sample_point))
                      * self.max_screening - self.sigma_prime(new_sample_point)))
                curr_sample_point = new_sample_point
            if T > 100:
                print(T)
            return ((np.sqrt(self.diffusion(*bdr_point))*T*self.next_flight(bdr_point, max_walk_length-1) + S)
                    / np.sqrt(self.diffusion(*point_to_check)))


    def find_pde(self):
        start_time = time.time()
        result = np.zeros(len(self.points_to_check))
        #derivatives = []
        CDF_hash = self.CDF()
        CDF_hash = CDF_hash/np.max(CDF_hash)
        match self.method:
            case "delta_tracking_recursion":
                for i, point_to_check in enumerate(self.points_to_check):
                    for walk_num in range(self.num_walks):
                        result[i] += self.delta_tracking_recursion(point_to_check, self.max_walk_length, CDF_hash)/self.num_walks
            case "next_flight":
                for i, point_to_check in enumerate(self.points_to_check):
                    for walk_num in range(self.num_walks):
                        result[i] += self.next_flight(point_to_check, self.max_walk_length, CDF_hash) / self.num_walks
            case "background_values":
                for i, point_to_check in enumerate(self.points_to_check):
                    result[i] = self.geometry.value_at_background(point_to_check)
            case "screening_coefficient":
                for i, point_to_check in enumerate(self.points_to_check):
                    result[i] = self.screening_coeff(point_to_check)
            case "diffusion":
                for i, point_to_check in enumerate(self.points_to_check):
                    result[i] = self.diffusion(*point_to_check)
            case "laplacian_diffusion":
                for i, point_to_check in enumerate(self.points_to_check):
                    result[i] = self.laplacian_diffusion(*point_to_check)
            case "norm_gradient_log_diffusion":
                for i, point_to_check in enumerate(self.points_to_check):
                    result[i] = self.norm_gradient_log_diffusion(*point_to_check)
            case _:
                pass
        print("time:", time.time() - start_time)
        print(f"walks*pixels = {self.geometry.bdr_max ** 2 * self.num_walks}")
        return result