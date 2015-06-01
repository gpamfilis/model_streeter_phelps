__author__ = 'George Pamfilis'
__version__ = '1.0'
__contact__ = 'gpamfilis@gmail.com'


import numpy as np
from numpy.random import ranf
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
        self.Csat = self.constants['csat'][0]
        self.C0 = self.constants['c0'][0]
        self.D0 = self.constants['csat'] - self.constants['c0']
        self.L0 = self.constants['L0'][0]  # BOD
        self.Lb = 4.0
        self.L = self.data['L-BOD'].values  # BOD
        self.log_L = np.log(self.L)
        self.time = self.distance_meters/(self.uav*60*60*24)
        # self.uav
        self.DO = self.data['DO'].values
        # plt.scatter(distance_meters,log_L)
        self.kd = -linregress(self.distance_meters, self.log_L)[0]*self.uav*60*60*24

    @staticmethod
    def general_graph(x, y, title, xlabel, ylabel):
        plt.plot(x,y,'.')
        plt.title(title)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def graph_two_lines(x1, y1, x2, y2):
        plt.plot(x1,y1,label = "Optimum Model Curve")
        plt.plot(x2,y2,'.',label= "Field Data Points")
        plt.title("Concentration of D.O. (mg/L) versus Distance (meters)")
        plt.grid(True)
        plt.xlabel("Distance (meters)")
        plt.ylabel("DO (mg/L)")
        plt.legend(loc = "upper right")
        plt.show()

    @staticmethod
    def root_mean_square_error(cf, cm):  # root mean square error
        n = len(cf)
        differences = []
        squared_difference = []
        for i in range(len(cf)):
            dif = cf[i] - cm[i]
            differences.append(dif)
        for i in range(len(cf)):
            squared_difference.append(differences[i]**2)
        sum_of_differences = np.sum(squared_difference)
        rmse = np.sqrt(sum_of_differences/n)
        return rmse

    @staticmethod
    def streeter_phelps(Csat, C0, L0, ka, kd, time):  # time is a 1-D array or list. #Streeter-Phelps

        C = lambda t: Csat - ((Csat - C0)*np.exp(-ka*t))-((kd*L0)/(ka-kd))*(np.exp(-kd*t)-np.exp(-ka*t))
        concentrations = np.zeros(len(time))
        for i, t in enumerate(time):
            concentrations[i] = C(t)
        return concentrations

    def find_ka_given_kd(self, number_of_trials=1000):
        kas = np.linspace(0., 1, num=number_of_trials)
        RMSES = np.zeros(number_of_trials)
        for i in range(number_of_trials):
            RMSES[i] = self.root_mean_square_error(self.DO, self.streeter_phelps(self.Csat, self.C0, self.L0,
                                                                                 kas[i], self.kd, self.time))
        # find the the minimum error in the RMSES array. find the index
        # (location) of that value. apply that location to kas list to find the
        # optimum ka with the smallest error.

        kamin1 = kas[np.where(RMSES == np.min(RMSES))[0]]  # minimum ka.
        print "the minimum ka is: ", kamin1, "and the corresponding root_mean_square_error is: ", np.min(RMSES)
        cm = self.streeter_phelps(self.Csat, self.C0, self.L0, 0.2032, self.kd, self.time)
        self.general_graph(self.distance_meters,cm,title=None,xlabel=None,ylabel=None)
        # this will store the model data using the ka with the least error
        # (with kd = 0.23)

    def fit_models_to_data_ka_and_kd_unknown(self):
        values = 1000  # number of points in each array
        kas2 = np.zeros(values)
        kds2 = np.zeros(values)
        RMSE2 = np.zeros(values)
        for i in range(values):
            if i % values == 500:
                print i
            kas2[i] = ranf()
            kds2[i] = ranf()
            RMSE2[i] = self.root_mean_square_error(self.DO, self.streeter_phelps(self.Csat, self.C0, self.L0,
                                                                                 kas2[i], kds2[i], self.time))
        itemindex = np.where(RMSE2 == np.min(RMSE2))
        kamin = kas2[itemindex[0][0]]
        kdmin = kds2[itemindex[0][0]]
        print "The smallest root_mean_square_error was: ", np.min(RMSE2), " for ka equal to: ", kamin, \
            " and kd equal to: ", kdmin
        coptimum = self.streeter_phelps(self.Csat, self.C0, self.L0, kamin, kdmin, self.time)

        self.graph_two_lines(self.distance_meters, self.DO, self.distance_meters, coptimum)

    @staticmethod
    def scatterplot3D(x, y, z, title, xlabel, ylabel, zlabel):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='g', marker='.')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.show()

    @staticmethod
    def triangle3dplot(x,y,z):

        #x = kas2
        #y = kds2
        #z = RMSE2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("root_mean_square_error versus ka and kd")
        ax.set_xlabel('ka (1/day)')
        ax.set_ylabel('kd (1/day)')
        ax.set_zlabel('root_mean_square_error')
        ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
        plt.show()

    @staticmethod
    def graphtwolines(x1,y1,x2,y2):
        plt.plot(x1,y1,label = "Optimum Model Curve")
        plt.plot(x2,y2,'.',label= "Field Data Points")
        plt.title("Concentration of D.O. (mg/L) versus Distance (meters)")
        plt.grid(True)
        plt.xlabel("Distance (meters)")
        plt.ylabel("DO (mg/L)")
        plt.legend(loc = "upper right")
        plt.show()


if __name__ == '__main__':
    sp = StreeterPhelps()
    sp.find_ka_given_kd()
    sp.fit_models_to_data_ka_and_kd_unknown()


