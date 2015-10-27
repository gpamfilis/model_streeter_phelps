__author__ = 'gpamfilis'
__version__ = '1.0'
__contact__ = 'gpamfilis@gmail.com'


import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit


class StreeterPhelps(object):

    def __init__(self):
        self.data = pd.read_excel('Data/Streeter_Phelps_input.xlsx', sheetname='data')
        self.constants = pd.read_excel('Data/Streeter_Phelps_input.xlsx', sheetname='constants')
        self.distance_miles = self.data['distance'].values
        for i in range(self.distance_miles.shape[0]-1):
            self.distance_miles[i+1] = self.distance_miles[i+1] - self.distance_miles[0]
        self.distance_miles[0] = 0
        self.distance_meters = self.distance_miles * 1.61*1000
        self.uav = self.constants['u'][0]  # m/s
        self.c_sat = self.constants['csat'][0]
        self.C0 = self.constants['c0'][0]
        self.D0 = self.constants['csat'] - self.constants['c0']
        self.L0 = self.constants['L0'][0]  # BOD
        self.Lb = 4.0
        self.L = self.data['L-BOD'].values  # BOD
        self.log_L = np.log(self.L)
        self.time = self.distance_meters/(self.uav*60*60*24)
        self.DO = self.data['DO'].values
        self.kd = -linregress(self.distance_meters, self.log_L)[0]*self.uav*60*60*24

    @staticmethod
    def xc_critical_distance(ka, kd, l0, d0, u):
        """
        :param ka:
        :param kd:
        :param l0:
        :param d0:
        :param u:
        :return:
        """
        xc = (u / (ka - kd)) * (np.log((ka / kd) * (1 - ((ka * kd) / kd) * (d0 / l0))))
        print("the critical distance is: {} meters".format(xc))

    def streeter_phelps(self, time, ka, kd):  # time is a 1-D array or list. #Streeter-Phelps
        """
        :param c_saturation:
        :param c0:
        :param l0:
        :param ka:
        :param kd:
        :param time:
        :return:
        """
        c = lambda t: self.c_sat - ((self.c_sat - self.C0) * np.exp(-ka * t)) - ((kd * self.L0)/(ka - kd)) * \
                                                                               (np.exp(-kd * t) - np.exp(-ka * t))
        return c(time)

    def find_ka_and_kd(self, initial_value_list=[0.3, 0.2]):
        xdata = self.time
        ydata = self.DO
        popt, pcov = curve_fit(StreeterPhelps().streeter_phelps, xdata, ydata, p0=initial_value_list)
        return popt, pcov

if __name__ == '__main__':
    pass

