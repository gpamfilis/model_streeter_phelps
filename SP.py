__author__ = 'George Pamfilis'
__version__ = '1.0'
__contact__ = 'gpamfilis@gmail.com'


import numpy as np
import pandas as pd
from scipy.stats import linregress

data = pd.read_excel('Data/Streeter_Phelps_input.xlsx', sheetname='data')
constants = pd.read_excel('Data/Streeter_Phelps_input.xlsx', sheetname='constants')

print data['L-BOD']
for i in range(10):
    print i, data['L-BOD'][i]
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
# plt.scatter(distance_meters,log_L)
a = linregress(distance_meters, log_L)
print(a[0]*uav*60*60*24)


