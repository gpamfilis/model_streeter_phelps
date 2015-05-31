__author__ = 'George Pamfilis'
__version__ = '1.0'
__contact__ = 'gpamfilis@gmail.com'


import numpy as np
from numpy.random import ranf
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

data = pd.read_excel('Data/Streeter_Phelps_input.xlsx', sheetname='data')
constants = pd.read_excel('Data/Streeter_Phelps_input.xlsx', sheetname='constants')

distance_miles = data['distance'].values
for i in range(distance_miles.shape[0]-1):
    distance_miles[i+1] = distance_miles[i+1] - distance_miles[0]
distance_miles[0] = 0
distance_meters = distance_miles * 1.61*1000

uav = constants['u'][0]  # m/s
Csat = constants['csat'][0]
C0 = constants['c0'][0]
D0 = constants['csat'] - constants['c0']
L0 = constants['L0'][0]  # BOD
Lb = 4.0
L = data['L-BOD'].values  # BOD
log_L = np.log(L)
time = distance_meters/uav
DO = data['DO'].values
# plt.scatter(distance_meters,log_L)
kd = -linregress(distance_meters, log_L)[0]*uav*60*60*24

def generalgraph(x, y, title, xlabel, ylabel):
    plt.plot(x,y,'.')
    plt.title(title)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
def graphtwolines(x1, y1, x2, y2):
    plt.plot(x1,y1,label = "Optimum Model Curve")
    plt.plot(x2,y2,'.',label= "Field Data Points")
    plt.title("Concentration of D.O. (mg/L) versus Distance (meters)")
    plt.grid(True)
    plt.xlabel("Distance (meters)")
    plt.ylabel("DO (mg/L)")
    plt.legend(loc = "upper right")
    plt.show()
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

def streeter_phelps(Csat, C0, L0, ka, kd, time):  # time is a 1-D array or list. #Streeter-Phelps
    # this is the streeter-phelps equation
    C = lambda t: Csat - ((Csat - C0)*np.exp(-ka*t))-((kd*L0)/(ka-kd))*(np.exp(-kd*t)-np.exp(-ka*t))
    concentrations = np.zeros(len(time))
    for i in range(len(time)):
        concentrations[i] = C(time[i])
    return concentrations

number_of_trials = 100
kas = np.linspace(0., 1, num=number_of_trials)

RMSES = np.zeros(number_of_trials)


for i in range(number_of_trials):
    RMSES[i] = root_mean_square_error(DO, streeter_phelps(Csat, C0, L0, kas[i], kd, time))
# find the the minimum error in the RMSES array. find the index
# (location) of that value. apply that location to kas list to find the
# optimum ka with the smallest error.

# kamin1 = kas[np.where(RMSES == np.min(RMSES))[0]]  # minimum ka.
print(np.where(RMSES == 0))


#print "the minimum ka is: ", kamin1, "and the corresponding root_mean_square_error is: ", np.min(RMSES)


'''
cm = streeter_phelps(Csat, C0, L0, kamin1, kd, time) #this will store the model data using the ka with the least error (with kd = 0.23)
values = 100000 #number of points in each array
kas2 = np.zeros(values)
kds2 = np.zeros(values)
RMSE2 = np.zeros(values)

for i in range(values):
    kas2[i] = ranf()
    kds2[i] = ranf()
    RMSE2[i] = root_mean_square_error(DO,streeter_phelps(Csat, C0, L0, kas2[i], kds2[i], time))
def scatterplot3D(x,y,z,title,xlabel,ylabel,zlabel):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(x,y,z,c = 'g',marker = '.')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def triangle3dplot():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    x = kas2
    y = kds2
    z = RMSE2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("root_mean_square_error versus ka and kd")
    ax.set_xlabel('ka (1/day)')
    ax.set_ylabel('kd (1/day)')
    ax.set_zlabel('root_mean_square_error')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    plt.show()

#find the the minimum error in the RMSE2 list. find the index (location) of that value. apply that location to kas list to find the
#optimum ka and kd with the smalest error.

itemindex = np.where(RMSE2 == np.min(RMSE2))
kamin = kas2[itemindex[0][0]]
kdmin = kds2[itemindex[0][0]]
print "The smallest root_mean_square_error was: ",np.min(RMSE2)," for ka equal to: ",kamin," and kd equal to: ",kdmin
coptimum = streeter_phelps(Csat, C0, L0, kamin, kdmin, time)
def graphtwolines(x1,y1,x2,y2):
    plt.plot(x1,y1,label = "Optimum Model Curve")
    plt.plot(x2,y2,'.',label= "Field Data Points")
    plt.title("Concentration of D.O. (mg/L) versus Distance (meters)")
    plt.grid(True)
    plt.xlabel("Distance (meters)")
    plt.ylabel("DO (mg/L)")
    plt.legend(loc = "upper right")
    plt.show()
data1 = np.array([kas,RMSES]).T
data2 = np.array([kas2,kds2,RMSE2]).T
'''