import numpy as np


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