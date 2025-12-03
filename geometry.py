import numpy as np
from matplotlib import pyplot as plt


class Square2D:
    def __init__(self, bdr_max, boundary_condition_func, background_value):
        self.name = 'Square2D'
        self.bdr_max = bdr_max
        self.boundary = np.array([[0, bdr_max], [bdr_max, bdr_max], [bdr_max, 0], [0, 0]])
        self.boundary_segments = np.array([[self.boundary[n - 1], self.boundary[n]] for n in range(len(self.boundary))])
        self.boundary_conditions = boundary_condition_func
        self.background_value = background_value
        self.shape = (bdr_max, bdr_max)

    def value_at_boundary(self, point):
        return self.boundary_conditions(point)

    def value_at_background(self, point):
        return self.background_value(point)

    def closest_boundary_point(self, current_point):
        R = np.inf
        x = np.array(current_point)
        closest_point = None
        rtest = np.inf
        for segment in self.boundary_segments:
            p = np.array(segment[0])
            q = np.array(segment[1])
            t = ((x - p) @ (q - p)) / (np.linalg.norm(p - q) ** 2)
            if t < 0:
                rtest = np.linalg.norm(x - p)
                if rtest < R:
                    closest_point = p
            elif t > 1:
                rtest = np.linalg.norm(x - q)
                if rtest < R:
                    closest_point = q
            else:
                rtest = np.linalg.norm(x - ((1 - t) * p + t * q))
                if rtest < R:
                    closest_point = (1 - t) * p + t * q
            R = min(rtest, R)
        return closest_point

    def points_to_check(self):
        return np.array([[k % self.bdr_max, k // self.bdr_max] for k in range(self.bdr_max ** 2)])


class Rectangle2D:
    def __init__(self, bdr_max_x, bdr_max_y, boundary_condition_func, background_value, boundary_segments=None):
        self.name = 'Square2D'
        self.bdr_max_x = bdr_max_x
        self.bdr_max_y = bdr_max_y
        self.boundary = np.array([[0, bdr_max_x], [bdr_max_x, bdr_max_y], [bdr_max_y, 0], [0, 0]])
        self.boundary_segments = boundary_segments
        if self.boundary_segments is None:
            self.boundary_segments = np.array([[self.boundary[n - 1], self.boundary[n]] for n in range(len(self.boundary))])
        self.boundary_conditions = boundary_condition_func
        self.background_value = background_value
        self.shape = (bdr_max_x, bdr_max_y)

    def value_at_boundary(self, point):
        return self.boundary_conditions(point)

    def value_at_background(self, point):
        return self.background_value(point)

    def closest_boundary_point(self, current_point):
        R = np.inf
        x = np.array(current_point)
        closest_point = None
        rtest = np.inf
        for segment in self.boundary_segments:
            p = np.array(segment[0])
            q = np.array(segment[1])
            t = ((x - p) @ (q - p)) / (np.linalg.norm(p - q) ** 2)
            if t < 0:
                rtest = np.linalg.norm(x - p)
                if rtest < R:
                    closest_point = p
            elif t > 1:
                rtest = np.linalg.norm(x - q)
                if rtest < R:
                    closest_point = q
            else:
                rtest = np.linalg.norm(x - ((1 - t) * p + t * q))
                if rtest < R:
                    closest_point = (1 - t) * p + t * q
            R = min(rtest, R)
        return closest_point

    def points_to_check(self):
        return np.array([[k % self.bdr_max_x, k // self.bdr_max_x] for k in range(self.bdr_max_x*self.bdr_max_y)])


class ParametricHalfSphere:
    def __init__(self, sample_num, boundary_segments=None):
        self.sample_num = sample_num
        self.bdr_max_x = 2*np.pi
        self.bdr_max_y = np.pi/2
        self.boundary_segments = boundary_segments
        if self.boundary_segments is None:
            self.boundary_segments = np.array([[[0, 0], [self.bdr_max_x, 0]]])

    def value_at_boundary(self, point):
        return np.sin(point[0])

    def value_at_background(self, point):
        return 0

    def closest_boundary_point(self, current_point):
        x = np.array(current_point)
        R = np.inf
        closest_point = None
        for segment in self.boundary_segments:
            p = np.array(segment[0])
            q = np.array(segment[1])
            t = ((x - p) @ (q - p)) / (np.linalg.norm(p - q) ** 2)
            if t < 0:
                rtest = np.linalg.norm(x - p)
                if rtest < R:
                    closest_point = p
            elif t > 1:
                rtest = np.linalg.norm(x - q)
                if rtest < R:
                    closest_point = q
            else:
                rtest = np.linalg.norm(x - ((1 - t) * p + t * q))
                if rtest < R:
                    closest_point = (1 - t) * p + t * q
            R = min(rtest, R)
        return closest_point

    def points_to_check(self):
        points_angle = []
        golden_ratio_rad = np.pi * (np.sqrt(5.) - 1.)
        for i in range(self.sample_num):
            angle_v = np.arcsin(1 - i/float(self.sample_num - 1))
            angle_u = (golden_ratio_rad * i) % self.bdr_max_x
            points_angle.append(np.array([angle_u, angle_v]))
        return points_angle

class ParametricHalfSphereConformal:
    def __init__(self, sample_num, radius):
        self.sample_num = sample_num
        self.radius = radius
        self.bdr_max = np.sqrt(sample_num)

    def value_at_boundary(self, point):
        return 0

    def value_at_background(self, point):
        value = 0
        norm = np.sqrt(point[0]**2 + point[1]**2)
        #if point[0]**2 < 0.3**2:
        #    if point[1]**2 < 0.3**2:
        #        value = 10
        if norm < 0.3:
            value = 10
        return value

    def closest_boundary_point(self, current_point):
        norm_length = np.sqrt(current_point[0]**2 + current_point[1]**2)
        if norm_length == 0:
            current_point += np.array([np.random.random(), np.random.random()])
            print('low vec found new = ', current_point)
        closest_boundry = self.radius * current_point / norm_length
        return closest_boundry

    def diffusion(self, u, v):
        return 1 / (4 / ((1 + u ** 2 + v ** 2) ** 2))

    def to_3D(self, points):
        u = points[0]
        v = points[1]
        length2 = np.square(u) + np.square(v)
        x = 2 * u / (1 + length2)
        y = 2 * v / (1 + length2)
        z = (1 - length2) / (1 + length2)
        return np.array([x, y, z])

    def points_to_check(self):
        golden_ratio_rad = np.pi * (np.sqrt(5.) + 1.)
        indices = np.arange(0, self.sample_num)
        radius_scaling = np.sqrt(indices/self.sample_num) - 0.001
        angle = golden_ratio_rad * indices

        u = radius_scaling * np.cos(angle)
        v = radius_scaling * np.sin(angle)

        #plt.scatter(u, v, 10 * self.diffusion(u, v))
        #plt.savefig('circle_in_2d.png')
        return np.array([u, v]).swapaxes(0, 1)
