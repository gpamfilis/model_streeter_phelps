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
    k_n_df = pd.DataFrame([k_n[0][0], k_n[0][1]]).T
    k_n_df.columns = ['ka', 'kb']
    k_n_df.to_csv('Data/ka_kb_parameters.txt', index=None)
    del sp, k_n, k_n_df

