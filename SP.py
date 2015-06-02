__author__ = 'George Pamfilis'
__version__ = '1.0'
__contact__ = 'gpamfilis@gmail.com'

import numpy as np
import numpy.random
import pandas as pd
from scipy.stats import linregress
from graphs import graph_two_lines, general_graph


class StreeterPhelps(object):

    def __init__(self):
        """

        :rtype : object
        """
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
        general_graph(self.distance_meters, self.log_L)
        self.kd = -linregress(self.distance_meters, self.log_L)[0]*self.uav*60*60*24

    @staticmethod
    def root_mean_square_error(experimental_data, model_data):
        """
        :param experimental_data:
        :param model_data:
        :return:
        """

        rmse = np.sqrt(np.sum((experimental_data - model_data)**2) / experimental_data.shape[0])
        return rmse

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
        print "the critical distance is: {} meters".format(xc)

    @staticmethod
    def streeter_phelps(c_saturation, c0, l0, ka, kd, time):  # time is a 1-D array or list. #Streeter-Phelps
        """
        :param c_saturation:
        :param c0:
        :param l0:
        :param ka:
        :param kd:
        :param time:
        :return:
        """
        c = lambda t: c_saturation - ((c_saturation - c0) * np.exp(-ka * t)) - ((kd * l0)/(ka - kd)) * \
                                                                               (np.exp(-kd * t) - np.exp(-ka * t))
        return c(time)

    def find_ka_given_kd(self, number_of_trials=1000):
        """
        :param number_of_trials:
        :return:
        """
        ka_array = np.linspace(0., 1, num=number_of_trials)
        root_error_array = np.zeros(number_of_trials)
        for i in range(number_of_trials):
            root_error_array[i] = self.root_mean_square_error(self.DO,
                                                              self.streeter_phelps(self.Csat, self.C0, self.L0,
                                                                                   ka_array[i], self.kd, self.time))
        ka_optimum_given_kd = ka_array[np.where(root_error_array == np.min(root_error_array))[0]]  # minimum ka.
        print "the minimum ka is: ", ka_optimum_given_kd, "and the corresponding root_mean_square_error is: ", \
            np.min(root_error_array)
        cm = self.streeter_phelps(self.Csat, self.C0, self.L0, ka_optimum_given_kd, self.kd, self.time)
        graph_two_lines(self.distance_meters, cm, self.distance_meters, self.DO)
        # this will store the model data using the ka with the least error
        # (with kd = 0.23)

    def fit_models_to_data_ka_and_kd_unknown(self, runs=10000):
        ka_available = np.zeros(runs)
        kd_available = np.zeros(runs)
        root_s_error = np.zeros(runs)
        for i in range(runs):
            ka_available[i] = numpy.random.ranf()
            kd_available[i] = numpy.random.ranf()
            root_s_error[i] = self.root_mean_square_error(self.DO, self.streeter_phelps(self.Csat,
                                                                                        self.C0, self.L0,
                                                                                        ka_available[i],
                                                                                        kd_available[i],
                                                                                        self.time))
        location_of_min_error = np.where(root_s_error == np.min(root_s_error))
        ka_min = ka_available[location_of_min_error[0][0]]
        kd_min = kd_available[location_of_min_error[0][0]]
        print "The smallest root_mean_square_error was: ", np.min(root_s_error), " for ka equal to: ", ka_min, \
            " and kd equal to: ", kd_min
        self.xc_critical_distance(ka_min, kd_min, self.L0, self.D0, self.uav*60*60*24)
        optimum_model_data = self.streeter_phelps(self.Csat, self.C0, self.L0, ka_min, kd_min, self.time)

        graph_two_lines(self.distance_meters, optimum_model_data, self.distance_meters, self.DO)

if __name__ == '__main__':
    sp = StreeterPhelps()
    sp.find_ka_given_kd()
    sp.fit_models_to_data_ka_and_kd_unknown()