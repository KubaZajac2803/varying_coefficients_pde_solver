import numpy as np
from scipy import special, optimize
import copy
import time
import sympy as sym
CDF_SAMPLES = 1000

class MonteCarloPDE2D:
    def __init__(self, geometry, num_walks, epsilon, max_walk_length, method, diffusion=lambda x, y: 0, laplacian_diffusion=lambda x, y: 0, norm_gradient_log_diffusion=lambda x, y:0, screening_coeff=lambda x: 0):
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
        max_sigma_bar = self.sigma_prime(
            optimize.minimize(lambda x: -1*self.sigma_prime(x),
                              x0=np.array([0, 0]),
                              method='trust-constr',
                              constraints=(optimize.NonlinearConstraint(lambda x: x[0], 0, self.geometry.bdr_max),
                                           optimize.NonlinearConstraint(lambda x: x[1], 0, self.geometry.bdr_max))).x)
        min_sigma_bar = self.sigma_prime(
            optimize.minimize(lambda x: self.sigma_prime(x),
                              x0=np.array([0, 0]),
                              method='trust-constr',
                              constraints=(optimize.NonlinearConstraint(lambda x: x[0], 0, self.geometry.bdr_max),
                                           optimize.NonlinearConstraint(lambda x: x[1], 0, self.geometry.bdr_max))).x)
        self.max_screening = max_sigma_bar - min_sigma_bar
        self.inv_cdf = self.inv_CDF()
    
    def sigma_prime(self, inside_point):
        return (self.screening_coeff(inside_point) / self.diffusion(*inside_point) +
                (self.laplacian_diffusion(*inside_point)/self.diffusion(*inside_point) -
                 (self.norm_gradient_log_diffusion(*inside_point)**2)/2) / 2)


    def Greens_2D(self, ball_radius, r):
         return (1/(2*np.pi))*(special.k0(r * np.sqrt(self.max_screening))
                    - (special.i0(r * np.sqrt(self.max_screening))
                    * special.k0(ball_radius * np.sqrt(self.max_screening))
                    / special.i0(ball_radius * np.sqrt(self.max_screening))))

    def Greens_2D_integral(self, ball_radius):
        return (1/self.max_screening)*(1 - 1/(special.i0(ball_radius * np.sqrt(self.max_screening))))


    def inv_CDF(self):
        x = np.linspace(1/CDF_SAMPLES, 1, CDF_SAMPLES)
        y = np.array([self.Greens_2D(1, i) for i in x])
        cdf = np.zeros(len(x))
        for i in range(1, len(x)):
            cdf[i] = cdf[i - 1] + (y[i] + y[i - 1]) / (2 * CDF_SAMPLES)
        x = x[1:]
        cdf = cdf[1:]
        cdf = cdf/np.max(cdf)
        # now do LSQ on (y, x) point pairs
        fitting_fun = lambda t: np.array([1/(1.1-t), np.exp(t), np.exp(2*t), np.exp(3*t), np.exp(4*t)])
        A = np.array([fitting_fun(known_point) for known_point in cdf])
        c = np.linalg.solve(A.T@A, A.T@x)
        c = c/(fitting_fun(1)@c)
        return lambda t: fitting_fun(t)@c


    def delta_tracking_recursion(self, point_to_check, max_walk_length):
        point_to_check = np.array(point_to_check)
        closest_boundary_point = self.geometry.closest_boundary_point(point_to_check)
        ball_radius = np.linalg.norm(point_to_check - closest_boundary_point)

        if ball_radius <= self.epsilon or max_walk_length <= 0:
            return self.geometry.value_at_boundary(closest_boundary_point)
        else:
            mu = np.random.random()
            rand_radius = ball_radius * self.inv_cdf(np.random.random())
            rand_angle_2 = 2 * np.pi * np.random.random()
            inside_point = point_to_check + rand_radius * np.array([np.cos(rand_angle_2), np.sin(rand_angle_2)])
            source_term = self.Greens_2D_integral(rand_radius)/(np.sqrt(self.diffusion(*point_to_check) * self.diffusion(*inside_point)))*self.geometry.value_at_background(inside_point)
            if mu <= self.max_screening*self.Greens_2D_integral(ball_radius):
                sigma_prime = self.sigma_prime(inside_point)
                return source_term + (np.sqrt(self.diffusion(*inside_point)/self.diffusion(*point_to_check))
                        *(1-sigma_prime/self.max_screening)*self.delta_tracking_recursion(inside_point, max_walk_length-1))
            else:
                rand_angle_1 = 2 * np.pi * np.random.random()
                bdr_point = point_to_check + ball_radius * np.array([np.cos(rand_angle_1), np.sin(rand_angle_1)])
                return source_term + (np.sqrt(self.diffusion(*bdr_point)/self.diffusion(*point_to_check)) *
                        self.delta_tracking_recursion(bdr_point, max_walk_length-1))


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
            russian_roulette_probability = 1
            rand_angle_1 = 2 * np.pi * np.random.random()
            bdr_point = point_to_check + ball_radius * np.array([np.cos(rand_angle_1), np.sin(rand_angle_1)])
            curr_sample_point = np.array(copy.deepcopy(point_to_check))
            while True:
                T += (W * self.Off_centered_Poisson_2D(ball_radius, point_to_check, curr_sample_point, bdr_point)
                      * (2*np.pi*ball_radius))

                W /= russian_roulette_probability
                rand_radius = ball_radius * np.random.random()
                rand_angle_2 = 2 * np.pi * np.random.random()
                new_sample_point = point_to_check + rand_radius * np.array([np.cos(rand_angle_2), np.sin(rand_angle_2)])
                W *= (self.Off_centered_greens_2D(ball_radius, point_to_check, curr_sample_point, new_sample_point)
                      * (self.max_screening - self.sigma_prime(new_sample_point))*self.Greens_2D_integral(ball_radius)
                      / self.Greens_2D(ball_radius, np.linalg.norm(point_to_check - new_sample_point)))
                russian_roulette_probability = min(1, W)
                if russian_roulette_probability < np.random.random():
                    break

                S += (self.geometry.value_at_background(new_sample_point)*W
                      / (np.sqrt(self.diffusion(*new_sample_point))
                      * (self.max_screening - self.sigma_prime(new_sample_point))))
                curr_sample_point = new_sample_point
            return ((np.sqrt(self.diffusion(*bdr_point))*T*self.next_flight(bdr_point, max_walk_length-1) + S)
                    / np.sqrt(self.diffusion(*point_to_check)))

    def find_pde(self):
        start_time = time.time()
        result = np.zeros(len(self.points_to_check))
        match self.method:
            case "delta_tracking_recursion":
                for i, point_to_check in enumerate(self.points_to_check):
                    for walk_num in range(self.num_walks):
                        result[i] += self.delta_tracking_recursion(point_to_check, self.max_walk_length)/self.num_walks
            case "next_flight":
                for i, point_to_check in enumerate(self.points_to_check):
                    for walk_num in range(self.num_walks):
                        result[i] += self.next_flight(point_to_check, self.max_walk_length) / self.num_walks
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


class EuclideanBrownianMotion:
    def __init__(self, geometry, epsilon, starting_point, diffusion, time_step):
        self.geometry = geometry
        self.epsilon = epsilon
        self.starting_point = starting_point
        self.diffusion = diffusion
        self.time_step = time_step

    def move_euclidean(self, point, d_to_bdr):
        point = np.array(point)
        step_vector = np.random.normal(0, 1, size=2)
        dp = np.sqrt(2*self.time_step*self.diffusion(*point))*step_vector
        dp_len = np.linalg.norm(dp)
        if dp_len > d_to_bdr:
            dp /= dp_len
            dp *= d_to_bdr
        new_point = point + dp
        return new_point

    def perform_walk(self):
        positions = [self.starting_point]
        curr_position = np.array(self.starting_point)
        closest_boundary_point = self.geometry.closest_boundary_point(curr_position)
        d_to_bdr = np.linalg.norm(curr_position - closest_boundary_point)
        k = 0
        while d_to_bdr > self.epsilon:
            k+=1
            if k % 1000 == 0:
                pass
            curr_position = self.move_euclidean(curr_position, d_to_bdr)
            positions.append(curr_position)
            closest_boundary_point = self.geometry.closest_boundary_point(curr_position)
            d_to_bdr = np.linalg.norm(curr_position - closest_boundary_point)
        return np.array(positions)


class RiemannianBrownianMotion: #can compare directly with euclidean case as then g_ij is the identity matrix
    def __init__(self, geometry, epsilon, starting_point, max_walk_length, diffusion, time_step, parameterization):
        self.geometry = geometry
        self.epsilon = epsilon
        self.starting_point = starting_point # in [u, v] coordinates
        self.diffusion = diffusion
        self.time_step = time_step
        self.parameterization = parameterization
        self.max_walk_length = max_walk_length

    def du_Riemannian_components(self):
        u, v = sym.var('u v')
        sym.init_printing()
        J = self.parameterization.jacobian((u, v))
        g = sym.Matrix(J.T@J)
        inv_metric = g.inv()
        sqrt_det_metric = sym.sqrt(sym.det(g))
        sqrt_inv_metric = sym.sqrt(g.inv()).doit().factor(deep=True)
        du1 = 1/2 * 1/sqrt_det_metric * (sym.diff(sqrt_det_metric * inv_metric[0, 0], u)
                                         + sym.diff(sqrt_det_metric * inv_metric[0, 1], v))
        du2 = 1/2 * 1/sqrt_det_metric * (sym.diff(sqrt_det_metric * inv_metric[1, 0], u)
                                         + sym.diff(sqrt_det_metric * inv_metric[1, 1], v))
        return du1, du2, sqrt_inv_metric

    def move_riemannian(self, point, d_to_bdr, du1, du2, sqrt_inv_metric):
        u, v = sym.var('u v')
        point = np.array(point)

        du1 = sym.lambdify((u, v), du1)
        du2 = sym.lambdify((u, v), du2)
        sqrt_inv_metric = sym.lambdify((u, v), sqrt_inv_metric)

        du_Riemannian = np.array([du1(*point), du2(*point)])

        step_length = np.random.normal(0, 1, size=2)
        du_Euclidean = np.sqrt(2*self.time_step*self.diffusion(*point))*step_length
        du_Euclidean = sqrt_inv_metric(*point)@du_Euclidean

        du = self.time_step*du_Riemannian + du_Euclidean
        du_len = np.linalg.norm(du)
        if du_len > d_to_bdr:
            du /= du_len
            du *= d_to_bdr
        new_point = point + du
        return np.array(new_point)

    def perform_walk(self, split=False):
        positions_split = []
        positions = [self.starting_point]
        curr_position = np.array(self.starting_point)
        closest_boundary_point = self.geometry.closest_boundary_point(curr_position)
        d_to_bdr = np.linalg.norm(curr_position - closest_boundary_point)
        du1, du2, sqrt_inv_metric = self.du_Riemannian_components()
        k = 0
        while d_to_bdr > self.epsilon and k < self.max_walk_length:
            k += 1
            if k % 1000 == 0:
                pass
            curr_position = self.move_riemannian(curr_position, d_to_bdr, du1, du2, sqrt_inv_metric)
            if split is True and not (0 < curr_position[0] < self.geometry.bdr_max_x):
                positions_split.append(positions)
                positions = []
            curr_position = [curr_position[0] % self.geometry.bdr_max_x, self.geometry.bdr_max_y - abs(curr_position[1] - self.geometry.bdr_max_y)] #THINK ABOUT THIS HEBUAIHGHUIREUBAKHBFGDUHI
            closest_boundary_point = self.geometry.closest_boundary_point(curr_position)
            d_to_bdr = np.linalg.norm(curr_position - closest_boundary_point)
            positions.append(curr_position)
        positions_split.append(positions)
        if split is True:
            return positions_split
        return positions

class MonteCarloRiemannianPDE2D:
    def __init__(self, geometry, num_walks, epsilon, max_walk_length, diffusion, time_step, parameterization):
        self.geometry = geometry
        self.num_walks = num_walks
        self.epsilon = epsilon
        self.max_walk_length = max_walk_length
        self.points_to_check = geometry.points_to_check()
        self.diffusion = diffusion
        self.time_step = time_step
        self.parameterization = parameterization

    def laplace(self, point_to_check):
        point_to_check = np.array(point_to_check)
        closest_boundary_point = self.geometry.closest_boundary_point(point_to_check)
        dist_to_bdr = np.linalg.norm(point_to_check - closest_boundary_point)

        if dist_to_bdr <= self.epsilon or self.max_walk_length <= 0:
            return self.geometry.value_at_boundary(closest_boundary_point)
        BM = RiemannianBrownianMotion(self.geometry, self.epsilon, point_to_check, self.max_walk_length,
                                        self.diffusion, self.time_step, self.parameterization)
        walk = BM.perform_walk()
        closest_boundary_point = walk[-1]
        return self.geometry.value_at_boundary(closest_boundary_point)

    def find_pde(self):
        start_time = time.time()
        result = np.zeros(len(self.points_to_check))

        for i, point in enumerate(self.points_to_check):
            for walk in range(self.num_walks):
                result[i] += self.laplace(point)/self.num_walks

        print("time:", time.time() - start_time)
        print(f"walks*pixels = {self.geometry.sample_num * self.num_walks}")
        return result
