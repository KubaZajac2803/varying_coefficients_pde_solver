import matplotlib.pyplot as plt
import numpy as np
import datetime


def plot(values, geometry):
    values = np.reshape(values, shape=geometry.shape)
    plt.imshow(values, cmap='cool', origin='lower', interpolation='None', interpolation_stage='data')
    plt.colorbar()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #plt.savefig(now + ".eps", format ='eps', bbox_inches = 'tight', pad_inches = 0)
    plt.show()