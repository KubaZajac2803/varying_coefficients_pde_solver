import numpy as np


class Square2D:
    def __init__(self, x, y, bdr_max, boundary_condition_func, background_value):
        self.name = 'Square2D'
        self.x = x
        self.y = y
        self.bdr_max = bdr_max
        self.boundary = np.array([[x, y + bdr_max], [x + bdr_max, y + bdr_max], [x + bdr_max, y], [x, y]])
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