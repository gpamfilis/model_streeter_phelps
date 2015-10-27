__author__ = 'gpamfilis'
__version__ = '1.0'
__contact__ = 'gpamfilis@gmail.com'


from model_streeter_phelps import StreeterPhelps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sp = StreeterPhelps()
    k_n = sp.find_ka_and_kd()
    pd.DataFrame(k_n, columns=['ka', 'kb']).to_csv('ka_kb_parameters.txt')
    del sp, k_n

