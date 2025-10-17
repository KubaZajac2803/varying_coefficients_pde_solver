from geometry import Square2D
from renderer import MonteCarloPDE2D
import matplotlib.pyplot as plt
import numpy as np
import datetime


class Display:
    def __init__(self, values):
        self.values = values
    def plot(self):
        plt.imshow(self.values, cmap='cool', origin='lower', interpolation='None', interpolation_stage='data')
        plt.colorbar()
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(now +"eps", format = 'eps', bbox_inches = 'tight', pad_inches = 0)
        plt.show()