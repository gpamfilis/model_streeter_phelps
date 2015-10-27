__author__ = 'gpamfilis'

import numpy as np
from model_streeter_phelps import StreeterPhelps
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    k_n = np.zeros((1000, 2))
    sp = StreeterPhelps()
    k = sp.find_ka_and_kd()[0]
    k_n[:, 0] = np.random.normal(k[0], k[0]*0.25, 1000)
    k_n[:, 1] = np.random.normal(k[1], k[1]*0.25, 1000)
    pd.DataFrame(k_n, columns=['ka', 'kb']).to_csv('Data/ka_kb_monte_carlo.txt', index=None)
    del k_n, sp, k

    k_n = pd.read_csv('Data/ka_kb_monte_carlo.txt')
    DO_array = np.zeros((len(sp.time), 10))
    for i in range(10):
        DO_array[:, i] = sp.streeter_phelps(sp.time, k_n['ka'][i], k_n['kb'][i])
    print(DO_array)
    pd.DataFrame(DO_array).plot()
    plt.show()
    # print(sp.find_ka_and_kd())

