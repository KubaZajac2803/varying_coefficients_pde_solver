from geometry import Square2D
from renderer import MonteCarloPDE2D
import matplotlib.pyplot as plt
import numpy as np
import datetime

"""
square_geometry = Square2D(0, 0, bdr_max, f_1)
    renderer = MonteCarloPDE2D(square_geometry,30, 10**(-3), 100, 1, pde_type)
    values = renderer.find_pde()
    values = np.reshape(values, shape=square_geometry.shape)

    plt.imshow(values, cmap='cool', origin='lower', interpolation='None', interpolation_stage='data')
    plt.colorbar()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(now+pde_type+".eps", format='eps', bbox_inches='tight', pad_inches = 0)
    plt.show()
    """