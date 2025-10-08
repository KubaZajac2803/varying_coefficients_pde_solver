from geometry import Square2D
from renderer import MonteCarloPDE2D
import matplotlib.pyplot as plt
import numpy as np
import datetime


bdr_max = 20
num_walks = 20
epsilon = 10e-6
method = "delta_tracking"
def bdr_cond(x):
    if x[0]>bdr_max/2:
        return 1
    else:
        return 0

def source_term(x):
    if x[0] > bdr_max/4 and x[0] < 3*bdr_max/4 and x[1] < bdr_max/2 :
        return 1
    else:
        return 0

screening_scaling = 2 # = max(sigma(x))
def screening_coefficient(x):
    if x[0] > bdr_max/4 and x[0] < 3*bdr_max/4 and x[1] < bdr_max/2 :
        return 1*screening_scaling
    else:
        return 0

def diffusion_coefficient(x):
    stdev = 5
    return 1/(2*np.pi*stdev**2)*np.exp(-((x[0]-bdr_max/2)**2 + (x[1]-bdr_max/2)**2)/(2*stdev**2))

if __name__ == '__main__':
    pass
